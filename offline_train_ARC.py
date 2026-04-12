import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.amp import autocast
from utils.args import parse_args
from utils.distribution import init_distributed_mode
from utils.load_model import load_models
from utils.wandb_vis import grid_to_pil
import wandb
from src.ARC_loader import build_dataloaders, IGNORE_INDEX


def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def contrastive_loss(image_features, text_features, temperature=0.07):
    """
    image_features: [Batch, Dim] (模型的输出，不需要提前归一化)
    text_features:  [Batch, Dim] (CLIP编码的文本，不需要提前归一化)
    temperature:    温度系数，越小对负样本的区分度要求越苛刻，通常 0.07 或 0.1
    """

    # 1. 维度检查与调整 (处理你代码里的 squeeze 逻辑)
    if image_features.dim() == 3:
        image_features = image_features.squeeze(1)
    if text_features.dim() == 3:
        text_features = text_features.squeeze(1)

    # 2. 强制归一化 (此处必须做，确保点积等于余弦相似度)
    # 这里再做一次只是计算量微增，但保证数学正确性
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    # 3. 计算相似度矩阵 [Batch, Batch]
    # logits[i][j] 表示 第i张图 和 第j个文本 的相似度
    logits = torch.matmul(image_features, text_features.T) / temperature

    # 4. 构造标签
    # 第0张图应该匹配第0个文本，第1张图匹配第1个文本...
    labels = torch.arange(logits.size(0), device=logits.device)

    # 5. 双向计算 Loss (图找文 + 文找图)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)

    return (loss_i2t + loss_t2i) / 2

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    distributed: bool = False,
    resolution_factor: int = 1,
) -> Tuple[float, float, Dict]: # 返回类型微调，之前没有Dict
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    total_exact = 0
    total_examples = 0

    visualizations = {}
    dataset = getattr(loader, "dataset", None)
    # if dataset is not None and hasattr(dataset, "disable_translation"):
    dataset.disable_translation()
    dataset.disable_resolution_augmentation(fix_scale_factor=resolution_factor)

    # 用于存储特征
    all_features = []
    all_task_ids = []

    for batch in loader:
        inputs = batch["inputs"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        task_ids = batch["task_ids"].to(device)
        offsets = batch["offset"].to(device)
        scale_factors = batch["scale_factors"].to(device)
        raw_outputs = batch["raw_outputs"]

        # 提取 Support Set (RuleEncoder 必须)
        support_images = batch["support_images"].to(device)
        support_targets = batch["support_targets"].to(device)
        if support_images is not None: support_images = support_images.to(device)
        if support_targets is not None: support_targets = support_targets.to(device)

        # logits = model(inputs, task_ids, attention_mask=attention_mask)

        # 评估时不需要 return_features=True，也不需要算 LARC Loss
        # 只需要 logits 算准确率
        outputs = model(
            pixel_values=inputs,
            support_images=support_images,
            support_targets=support_targets,
            task_ids=task_ids,
            attention_mask=attention_mask,
            return_features=True  # 开启特征返回
        )

        # 兼容性处理：防止未来 model 返回 tuple
        if isinstance(outputs, tuple):
            logits = outputs[0]
            # 假设 outputs[1] 是 [Batch, Dim] 或 [Batch, 1, Dim] 的特征
            features = outputs[1]
        else:
            logits = outputs

        num_colors = logits.size(1)
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
        loss = F.cross_entropy(
            logits_flat,
            targets.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="sum",
        )
        total_loss += loss.item()
        total_pixels += (targets != IGNORE_INDEX).sum().item()

        predictions = logits.argmax(dim=1)
        batch_size = predictions.size(0)

        # 收集特征到 CPU 内存
        if save_features and features is not None:
            # 确保特征是 2D [Batch, Dim]
            if features.dim() == 3:
                features = features.squeeze(1)
            all_features.append(features.cpu())
            all_task_ids.append(task_ids.cpu())

        for idx in range(batch_size):
            target = targets[idx]
            prediction = predictions[idx]
            valid = target != IGNORE_INDEX
            if valid.any():
                is_exact = bool(torch.equal(prediction[valid], target[valid]))
            else:
                is_exact = False
            total_exact += int(is_exact)
            total_examples += 1

            input_grid = inputs[idx]
            mask = attention_mask[idx]
            visualizations[task_ids[idx].item()] = grid_to_pil(mask, input_grid, target, prediction, IGNORE_INDEX=IGNORE_INDEX)

        # --- 保存特征文件 ---
        if save_features and len(all_features) > 0:
            # 拼接所有 Batch
            all_features_cat = torch.cat(all_features, dim=0)
            all_task_ids_cat = torch.cat(all_task_ids, dim=0)

            # 如果是分布式，只让主进程保存
            if not distributed or (dist.is_initialized() and dist.get_rank() == 0):
                print(f"Saving features for t-SNE: {all_features_cat.shape} to {save_path}")
                torch.save({
                    "features": all_features_cat,
                    "task_ids": all_task_ids_cat
                }, save_path)

    if distributed and dist.is_initialized():
        totals = torch.tensor(
            [total_loss, total_pixels, total_exact, total_examples],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_pixels, total_exact, total_examples = totals.tolist()

    avg_loss = total_loss / max(total_pixels, 1)
    accuracy = total_exact / max(total_examples, 1)

    if not args.disable_translation:
        dataset.enable_translation()
    if not args.disable_resolution_augmentation:
        dataset.enable_resolution_augmentation()
    return avg_loss, accuracy, visualizations

def train(args: argparse.Namespace) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    set_seed(args.seed + (rank if distributed else 0))

    # build_dataloaders 调用
    # 传入 clip_embeddings_path 和 k_shot
    # 注意：确保 utils/args.py 里已经解析了这些参数，如果没有，可以在这里给默认值
    clip_path = getattr(args, "clip_embeddings_path", None)
    k_shot = getattr(args, "k_shot", 3)

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        text_embeddings_path=clip_path  # 传入路径
    )

    if args.disable_translation:
        train_dataset.disable_translation()
        if eval_dataset is not None:
            eval_dataset.disable_translation()
    else:
        train_dataset.enable_translation()
        if eval_dataset is not None:
            eval_dataset.enable_translation()

    if args.disable_resolution_augmentation:
        train_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
        if eval_dataset is not None:
            eval_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
    else:
        train_dataset.enable_resolution_augmentation()
        if eval_dataset is not None:
            eval_dataset.enable_resolution_augmentation()

    total_train_examples = len(train_dataset)

    if (not distributed) or rank == 0:
        print(f"Total training examples: {total_train_examples}")

    model, model_for_eval, optimizer, scaler, scheduler, start_epoch = load_models(
        args=args, train_dataset=train_dataset, device=device, distributed=distributed, rank=rank, local_rank=local_rank
    )
    autocast_device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cuda"

    wandb_run = None
    is_main_process = (not distributed) or rank == 0

    if args.use_wandb and is_main_process:
        if wandb is None:
            raise RuntimeError(
                "Weights & Biases is not installed. Install wandb or disable --use-wandb."
            )

        wandb_kwargs: Dict[str, Any] = {
            "project": args.wandb_project,
            "config": dict(vars(args)),
        }

        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name

        wandb_run = wandb.init(**wandb_kwargs)
        wandb.watch(model_for_eval, log=None)

    best_eval_acc = float("-inf")
    global_start = time.time()
    previous_total_steps = 0

    # 获取 LARC 权重，默认 0.1
    larc_weight = getattr(args, "larc_weight", 0.1)
    use_larc = getattr(args, "use_larc", False)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()
            train_exact = 0
            train_examples = 0

            for step, batch in enumerate(train_loader, 1):
                inputs = batch["inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                task_ids = batch["task_ids"].to(device)

                # 2. Support Set 和 Text Embedding
                support_images = batch["support_images"].to(device)
                support_targets = batch["support_targets"].to(device)
                text_gt = batch["description_embeddings"].to(device)  # [B, 512]

                optimizer.zero_grad(set_to_none=True)
                
                # Use automatic mixed precision
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    # logits = model(inputs, task_ids, attention_mask=attention_mask)
                    # 3. 模型前向传播
                    if use_larc:
                        # 开启 LARC：请求返回特征
                        logits, text_pred = model(
                            pixel_values=inputs,
                            support_images=support_images,
                            support_targets=support_targets,
                            task_ids=task_ids,
                            attention_mask=attention_mask,
                            return_features=True
                        )

                        # 4. 计算 ARC 主 Loss
                        num_colors = logits.size(1)
                        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
                        task_loss = F.cross_entropy(
                            logits_flat,
                            targets.view(-1),
                            ignore_index=IGNORE_INDEX,
                        )

                        # 5. 计算 LARC 对齐 Loss
                        if text_pred.dim() == 3:
                            text_pred = text_pred.squeeze(1)
                        # 使用MSE Loss
                        # text_pred_norm = F.normalize(text_pred, dim=-1)
                        # text_gt_norm = F.normalize(text_gt, dim=-1)
                        #
                        # larc_loss_val = F.mse_loss(text_pred_norm, text_gt_norm)
                        # 使用对比损失
                        larc_loss_val = contrastive_loss(text_pred, text_gt, temperature=0.07)

                        # 总 Loss
                        loss = task_loss + larc_weight * larc_loss_val

                    else:
                        # 关闭 LARC：常规训练
                        logits = model(
                            pixel_values=inputs,
                            support_images=support_images,
                            support_targets=support_targets,
                            task_ids=task_ids,
                            attention_mask=attention_mask
                        )
                        num_colors = logits.size(1)
                        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
                        loss = F.cross_entropy(
                            logits_flat,
                            targets.view(-1),
                            ignore_index=IGNORE_INDEX,
                        )
                        larc_loss_val = torch.tensor(0.0)

                batch_size = inputs.size(0)

                predictions = logits.argmax(dim=1)
                for idx in range(batch_size):
                    target = targets[idx]
                    prediction = predictions[idx]
                    valid = target != IGNORE_INDEX
                    if valid.any():
                        is_exact = bool(torch.equal(prediction[valid], target[valid]))
                    else:
                        is_exact = False
                    train_exact += int(is_exact)
                    train_examples += 1

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item() * batch_size
                sample_count += batch_size

                if total_batches > 0 and is_main_process and step % 10 == 0:  # Update every 10 steps
                    elapsed = time.time() - epoch_start
                    avg_step_time = elapsed / step
                    steps_completed = previous_total_steps + step
                    total_steps = len(train_loader) * args.epochs
                    remaining_steps = total_steps - steps_completed
                    elapsed_global = time.time() - global_start
                    avg_time_per_step_global = elapsed_global / max(steps_completed, 1)
                    eta = remaining_steps * avg_time_per_step_global
                    bar_length = 30
                    progress_ratio = steps_completed / total_steps if total_steps else 0
                    filled = int(bar_length * progress_ratio)
                    bar = "#" * filled + "-" * (bar_length - filled)
                    progress = 100.0 * progress_ratio
                    sys.stdout.write(
                        f"\rEpoch {epoch} [{bar}] {progress:5.1f}% ETA {_format_eta(eta)}"
                    )
                    sys.stdout.flush()

            if total_batches > 0 and is_main_process:
                sys.stdout.write("\n")
            previous_total_steps += total_batches

            epoch_duration = time.time() - epoch_start if total_batches > 0 else 0.0

            train_totals = torch.tensor(
                [running_loss, sample_count, train_exact, train_examples],
                dtype=torch.float64,
                device=device,
            )
            if distributed and dist.is_initialized():
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            running_loss_total, sample_count_total, train_exact_total, train_examples_total = train_totals.tolist()
            avg_train_loss = running_loss_total / max(sample_count_total, 1)
            train_acc = train_exact_total / max(train_examples_total, 1)

            total_elapsed = time.time() - global_start
            total_steps = len(train_loader) * args.epochs
            steps_completed = min(previous_total_steps, total_steps)
            remaining_steps = total_steps - steps_completed
            avg_time_per_step_global = total_elapsed / max(steps_completed, 1)
            total_eta = remaining_steps * avg_time_per_step_global

            log_parts = [
                f"epoch={epoch}",
                f"train_loss={avg_train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"epoch_time={epoch_duration:.1f}s",
                f"eta_total={_format_eta(total_eta)}",
            ]

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.learning_rate
            log_parts.append(f"lr={current_lr:.6f}")

            eval_loss = None
            eval_acc = None
            visualizations = {}
            run_eval = eval_loader is not None 
            if run_eval:
                # 1. 创建一个专门存放特征的目录，保持整洁
                tsne_dir = Path("saves/tsne_features")
                tsne_dir.mkdir(parents=True, exist_ok=True)

                # 2. 文件名带上 Epoch，例如: saves/tsne_features/features_epoch_0.pt
                current_save_path = tsne_dir / f"features_epoch_{epoch}.pt"

                eval_loss, eval_acc, visualizations = evaluate(
                    model,
                    eval_loader,
                    device,
                    distributed=distributed,
                    resolution_factor=args.fix_scale_factor if args.disable_resolution_augmentation else 2,
                    save_features=True,  # 显式开启保存
                    save_path=str(current_save_path)  # 传入动态生成的路径
                )
                if is_main_process:
                    log_parts.append(f"eval_loss={eval_loss:.4f}")
                    log_parts.append(f"eval_acc={eval_acc:.4f}")

                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        if args.best_save_path:
                            best_path = Path(args.best_save_path)
                            best_path.parent.mkdir(parents=True, exist_ok=True)
                            model_to_save = model_for_eval
                            best_payload: Dict[str, Any] = {
                                "model_state": model_to_save.state_dict(),
                                "config": vars(args),
                                "best_eval_accuracy": best_eval_acc,
                                "epoch": epoch,
                            }
                            if optimizer is not None:
                                best_payload["optimizer_state"] = optimizer.state_dict()
                            if scheduler is not None:
                                best_payload["scheduler_state"] = scheduler.state_dict()
                            if scaler.is_enabled():
                                best_payload["scaler_state"] = scaler.state_dict()
                            torch.save(best_payload, best_path)

            if is_main_process:
                print(" | ".join(log_parts))

            if wandb_run is not None and is_main_process:
                metrics = {
                    "epoch": epoch,
                    "steps": previous_total_steps,
                    "train/loss": avg_train_loss,
                    "train/accuracy": train_acc,
                    "train/epoch_time": epoch_duration,
                    "train/lr": current_lr,
                }
                if eval_loss is not None and eval_acc is not None:
                    metrics["eval/loss"] = eval_loss
                    metrics["eval/accuracy"] = eval_acc
                    if (epoch + 1) % args.vis_every == 0:
                        reverse_task_lookup = {v: k for k, v in eval_dataset.task_lookup.items()}
                        metrics["visualizations/eval"] = [wandb.Image(v, mode="RGBA", caption=f"task {reverse_task_lookup[k]}") for k, v in visualizations.items()]
                if best_eval_acc > float("-inf"):
                    metrics["eval/best_accuracy"] = best_eval_acc
                wandb.log(metrics, step=epoch)

            if scheduler is not None:
                scheduler.step()

    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if distributed and dist.is_initialized():
            dist.barrier()

    if args.save_path and is_main_process:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_payload = {"model_state": model_for_eval.state_dict(), "config": vars(args)}
        if scaler.is_enabled():
            final_payload["scaler_state"] = scaler.state_dict()
        torch.save(final_payload, save_path)

    if distributed and dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    args = parse_args()
    train(args)
