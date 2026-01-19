# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import math
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as video_vit
from app.frame_decoder.retro_dataset import init_retro_frame_data
from app.vjepa.transforms import make_transforms
from src.models.frame_decoder import FrameDecoderConfig, FrameDecoderViT, clip_to_unit_rgb
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule
from src.utils.wrappers import MultiSeqWrapper

logger = get_logger(__name__, force=True)

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def _strip_prefix_if_present(state_dict: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update dst with src, returning dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _build_frozen_encoder(enc_train_cfg: Dict[str, Any], device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    cfg_model = enc_train_cfg.get("model", {})
    cfg_data = enc_train_cfg.get("data", {})
    cfg_meta = enc_train_cfg.get("meta", {})

    model_name = cfg_model.get("model_name", "vit_large")
    crop_size = int(cfg_data.get("crop_size", 256))
    patch_size = int(cfg_data.get("patch_size", 16))
    tubelet_size = int(cfg_data.get("tubelet_size", 2))
    max_num_frames = int(max(cfg_data.get("dataset_fpcs", [cfg_data.get("frames_per_clip", 16)])))

    encoder_backbone = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=bool(cfg_model.get("uniform_power", False)),
        use_sdpa=bool(cfg_meta.get("use_sdpa", False)),
        use_silu=bool(cfg_model.get("use_silu", False)),
        wide_silu=bool(cfg_model.get("wide_silu", True)),
        use_activation_checkpointing=bool(cfg_model.get("use_activation_checkpointing", False)),
        use_rope=bool(cfg_model.get("use_rope", False)),
    )
    encoder = MultiSeqWrapper(encoder_backbone).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder, {"crop_size": crop_size, "patch_size": patch_size, "tubelet_size": tubelet_size}


def _build_decoder(args: Dict[str, Any], encoder_dim: int, crop_size: int, patch_size: int, use_sdpa: bool) -> FrameDecoderViT:
    cfg_dec = args.get("decoder", {}) or {}

    # Default to ViT-L-ish sizing (as described in Appendix B.3), but allow overrides.
    dec_cfg = FrameDecoderConfig(
        img_size=int(cfg_dec.get("img_size", crop_size)),
        patch_size=int(cfg_dec.get("patch_size", patch_size)),
        in_dim=int(cfg_dec.get("in_dim", encoder_dim)),
        embed_dim=int(cfg_dec.get("embed_dim", encoder_dim)),
        depth=int(cfg_dec.get("depth", 24)),
        num_heads=int(cfg_dec.get("num_heads", 16)),
        mlp_ratio=float(cfg_dec.get("mlp_ratio", 4.0)),
        out_chans=int(cfg_dec.get("out_chans", 3)),
        use_sdpa=bool(cfg_dec.get("use_sdpa", use_sdpa)),
        use_silu=bool(cfg_dec.get("use_silu", False)),
        wide_silu=bool(cfg_dec.get("wide_silu", True)),
        out_act=str(cfg_dec.get("out_act", "none")),
    )
    return FrameDecoderViT(dec_cfg)


def _init_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float,
    betas=(0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or p.ndim == 1:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    param_groups = [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=float(lr), betas=betas, eps=float(eps))


def main(args, resume_preempt: bool = False):
    folder = args.get("folder")
    if folder is None:
        raise ValueError("Missing required config key: folder")

    cfg_meta = args.get("meta", {}) or {}
    use_tensorboard = bool(cfg_meta.get("use_tensorboard", False))
    tensorboard_logdir = cfg_meta.get("tensorboard_logdir")
    tb_image_freq = int(cfg_meta.get("tensorboard_image_freq", 0) or 0)
    tb_num_images = int(cfg_meta.get("tensorboard_num_images", 4) or 4)

    dtype_name = str(cfg_meta.get("dtype", "bfloat16")).lower()
    if dtype_name == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif dtype_name == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    seed = int(cfg_meta.get("seed", _GLOBAL_SEED))
    np.random.seed(seed)
    torch.manual_seed(seed)

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    if not torch.cuda.is_available():
        device = torch.device("cpu")
        local_rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)

    writer = None
    if use_tensorboard and rank == 0:
        tb_dir = tensorboard_logdir or os.path.join(folder, "tensorboard")
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=tb_dir)
            logger.info(f"TensorBoard enabled: writing logs to {tb_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize TensorBoard SummaryWriter at {tb_dir}: {e}")
            writer = None

    # --------------------------------------------------------------------- #
    # Encoder config / checkpoint (re-using the encoder training config avoids
    # duplicating the huge dataset list in a decoder YAML).
    # --------------------------------------------------------------------- #
    cfg_encoder = args.get("encoder", {}) or {}
    enc_cfg_path = cfg_encoder.get("config")
    enc_ckpt_path = cfg_encoder.get("checkpoint")
    if enc_cfg_path is None or enc_ckpt_path is None:
        raise ValueError("decoder config must specify encoder.config and encoder.checkpoint")

    with open(enc_cfg_path, "r") as f:
        enc_train_cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Build frozen encoder from the encoder training cfg.
    encoder, enc_shape = _build_frozen_encoder(enc_train_cfg, device=device)
    crop_size = int(enc_shape["crop_size"])
    patch_size = int(enc_shape["patch_size"])
    tubelet_size = int(enc_shape["tubelet_size"])

    ckpt = robust_checkpoint_loader(enc_ckpt_path, map_location=torch.device("cpu"))
    enc_sd = ckpt.get("encoder")
    if enc_sd is None:
        raise ValueError(f"Checkpoint missing 'encoder' key: {enc_ckpt_path}")
    enc_sd = _strip_prefix_if_present(enc_sd, "module.")
    msg = encoder.load_state_dict(enc_sd, strict=False)
    logger.info(f"Loaded frozen encoder from {enc_ckpt_path} with msg: {msg}")
    del ckpt

    # --------------------------------------------------------------------- #
    # Data (defaults to encoder cfg, with decoder YAML overrides).
    # --------------------------------------------------------------------- #
    cfg_data = deepcopy(enc_train_cfg.get("data", {}) or {})
    _deep_update(cfg_data, args.get("data", {}) or {})

    dataset_type = str(cfg_data.get("dataset_type", "retro")).lower()
    if dataset_type != "retro":
        raise NotImplementedError("frame_decoder currently supports dataset_type=retro only")

    data_paths = cfg_data.get("datasets", [])
    manifest_paths = cfg_data.get("manifest_paths")
    action_mappings = cfg_data.get("action_mappings")
    state_keys = cfg_data.get("state_keys")
    action_dim = cfg_data.get("action_dim", None)
    # RetroGameDataset requires a global action dim when action_mappings are provided.
    # Encoder configs commonly store this under model.action_embed_dim.
    if action_dim is None:
        action_dim = (enc_train_cfg.get("model", {}) or {}).get("action_embed_dim", None)
    if action_dim is None and action_mappings:
        logger.warning("action_mappings provided but action_dim is None; disabling action_mappings for decoder training.")
        action_mappings = None
    frame_stride = int(cfg_data.get("frame_stride", 1) or 1)
    frames_per_clip = int(cfg_data.get("frames_per_clip", 4))
    batch_size = int(cfg_data.get("batch_size", 32))
    num_workers = int(cfg_data.get("num_workers", 8))
    pin_mem = bool(cfg_data.get("pin_mem", True))
    persistent_workers = bool(cfg_data.get("persistent_workers", True))
    manifest_cache = bool(cfg_data.get("manifest_cache", False))
    manifest_cache_dir = cfg_data.get("manifest_cache_dir")

    if frames_per_clip < tubelet_size or frames_per_clip % tubelet_size != 0:
        raise ValueError(f"frames_per_clip must be divisible by tubelet_size, got {frames_per_clip=}, {tubelet_size=}")

    # Data aug defaults from encoder cfg.
    cfg_aug = deepcopy(enc_train_cfg.get("data_aug", {}) or {})
    _deep_update(cfg_aug, args.get("data_aug", {}) or {})
    transform = make_transforms(
        random_horizontal_flip=bool(cfg_aug.get("horizontal_flip", False)),
        random_resize_aspect_ratio=tuple(cfg_aug.get("random_resize_aspect_ratio", (3 / 4, 4 / 3))),
        random_resize_scale=tuple(cfg_aug.get("random_resize_scale", (0.3, 1.0))),
        reprob=float(cfg_aug.get("reprob", 0.0)),
        auto_augment=bool(cfg_aug.get("auto_augment", False)),
        motion_shift=bool(cfg_aug.get("motion_shift", False)),
        crop_size=int(cfg_data.get("crop_size", crop_size)),
    )

    loader, sampler = init_retro_frame_data(
        data_paths=data_paths,
        batch_size=batch_size,
        frames_per_clip=frames_per_clip,
        frame_stride=frame_stride,
        transform=transform,
        manifest_paths=manifest_paths,
        action_mappings=action_mappings,
        action_dim=action_dim,
        state_keys=state_keys,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        world_size=world_size,
        rank=rank,
        drop_last=True,
        manifest_cache=manifest_cache,
        manifest_cache_dir=manifest_cache_dir,
    )

    # --------------------------------------------------------------------- #
    # Decoder + optimization (Appendix B.3 defaults, overridable).
    # --------------------------------------------------------------------- #
    cfg_opt = args.get("optimization", {}) or {}
    total_steps = int(cfg_opt.get("total_steps", 150_000))
    warmup_steps = int(cfg_opt.get("warmup_steps", 2000))
    lr = float(cfg_opt.get("lr", 5e-4))
    start_lr = float(cfg_opt.get("start_lr", 0.0))
    final_lr = float(cfg_opt.get("final_lr", 0.0))
    weight_decay = float(cfg_opt.get("weight_decay", 0.1))
    final_weight_decay = float(cfg_opt.get("final_weight_decay", weight_decay))
    grad_clip_norm = float(cfg_opt.get("grad_clip_norm", 1.0))
    betas = tuple(cfg_opt.get("betas", (0.9, 0.999)))
    eps = float(cfg_opt.get("eps", 1.0e-8))

    decoder = _build_decoder(args, encoder_dim=int(encoder.backbone.embed_dim), crop_size=crop_size, patch_size=patch_size, use_sdpa=bool(enc_train_cfg.get("meta", {}).get("use_sdpa", False)))
    decoder.to(device)
    decoder = DistributedDataParallel(decoder, static_graph=True)

    optimizer = _init_optimizer(decoder, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    lr_sched = WarmupCosineSchedule(
        optimizer,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        T_max=total_steps,
    )
    wd_sched = CosineWDSchedule(optimizer, ref_wd=weight_decay, final_wd=final_weight_decay, T_max=total_steps)

    use_grad_scaler = bool(cfg_meta.get("use_grad_scaler", True))
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision and use_grad_scaler and dtype == torch.float16)

    # --------------------------------------------------------------------- #
    # Checkpointing + logging
    # --------------------------------------------------------------------- #
    Path(folder).mkdir(parents=True, exist_ok=True)
    latest_path = os.path.join(folder, "latest.pt")
    log_path = os.path.join(folder, f"log_r{rank}.csv")

    save_freq = int(cfg_meta.get("save_freq", 1000))
    log_freq = int(cfg_meta.get("log_freq", 20))
    load_model = bool(cfg_meta.get("load_checkpoint", False)) or resume_preempt
    read_checkpoint = cfg_meta.get("read_checkpoint", None) or latest_path
    load_opt_state = bool(cfg_meta.get("load_optimizer", True))

    csv_logger = CSVLogger(
        log_path,
        ("%d", "step"),
        ("%.5f", "loss"),
        ("%.2e", "lr"),
        ("%.2e", "wd"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
    )

    start_step = 0
    if load_model and os.path.exists(read_checkpoint):
        logger.info(f"Loading decoder checkpoint from {read_checkpoint}")
        d = robust_checkpoint_loader(read_checkpoint, map_location=torch.device("cpu"))
        if "decoder" in d:
            msg = decoder.load_state_dict(d["decoder"], strict=False)
            logger.info(f"Loaded decoder weights with msg: {msg}")
        if load_opt_state and "opt" in d:
            try:
                optimizer.load_state_dict(d["opt"])
                if scaler is not None and d.get("scaler") is not None:
                    scaler.load_state_dict(d["scaler"])
            except Exception as exc:
                logger.warning(f"Failed to load optimizer/scaler state: {exc}")
        start_step = int(d.get("step", 0))
        # Advance schedulers to match step.
        for _ in range(start_step):
            lr_sched.step()
            wd_sched.step()
        del d

    def save_checkpoint(step: int, path: str) -> None:
        if rank != 0:
            return
        payload = {
            "decoder": decoder.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "step": int(step),
            "encoder_checkpoint": str(enc_ckpt_path),
            "encoder_config": str(enc_cfg_path),
        }
        try:
            torch.save(payload, path)
        except Exception as exc:
            logger.warning(f"Failed to save checkpoint to {path}: {exc}")

    # --------------------------------------------------------------------- #
    # Training loop (step-based, as in Appendix B.3)
    # --------------------------------------------------------------------- #
    sampler.set_epoch(0)
    it = iter(loader)
    epoch = 0
    loss_meter = AverageMeter()

    feature_norm = bool((args.get("decoder", {}) or {}).get("feature_layer_norm", True))

    for step in range(start_step, total_steps):
        t0 = time.time()

        try:
            clip = next(it)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            it = iter(loader)
            clip = next(it)

        # clip: [B, C, T, H, W] in encoder-normalized space
        clip = clip.to(device, non_blocking=True)
        data_elapsed_ms = (time.time() - t0) * 1000.0

        global_step = step + 1
        should_log_images = (
            writer is not None and tb_image_freq > 0 and (global_step % tb_image_freq == 0) and tb_num_images > 0
        )

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            new_lr = lr_sched.step()
            new_wd = wd_sched.step()

            with torch.no_grad():
                z = encoder([clip])[0]  # [B, N, D]
                if feature_norm:
                    z = F.layer_norm(z, (z.size(-1),))

            b, _, d = z.shape
            _, _, t, h, w = clip.shape
            t_tokens = t // tubelet_size
            h_p = h // patch_size
            w_p = w // patch_size
            n_per_frame = h_p * w_p
            expected_n = t_tokens * n_per_frame
            if z.size(1) != expected_n:
                raise ValueError(f"Unexpected token count: got N={z.size(1)}, expected {expected_n}")

            # Slice per-tubelet "frame" and decode independently.
            z = z.reshape(b, t_tokens, n_per_frame, d).reshape(b * t_tokens, n_per_frame, d)

            # Reconstruct the first frame in each tubelet (aligns with PatchEmbed3D stride).
            target = clip_to_unit_rgb(clip)[:, :, ::tubelet_size, :, :]  # [B, 3, t_tokens, H, W]
            target = target.permute(0, 2, 1, 3, 4).reshape(b * t_tokens, 3, h, w)

            with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                pred = decoder(z)  # [B*t_tokens, 3, H, W]
                loss = F.mse_loss(pred, target)

            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=grad_clip_norm)
                optimizer.step()

            vis = None
            if should_log_images:
                n = min(int(tb_num_images), int(pred.shape[0]))
                # Concatenate target and prediction for easier qualitative inspection.
                vis = torch.cat([target[:n], pred[:n]], dim=3).detach().float().cpu().clamp(0.0, 1.0)
            return float(loss.item()), float(new_lr), float(new_wd), vis

        (loss, new_lr, new_wd, vis), gpu_ms = gpu_timer(train_step)
        iter_ms = (time.time() - t0) * 1000.0

        loss_meter.update(loss)
        csv_logger.log(step + 1, loss, new_lr, new_wd, iter_ms, gpu_ms, data_elapsed_ms)

        if writer is not None:
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/loss_avg", loss_meter.avg, global_step)
            writer.add_scalar("train/lr", new_lr, global_step)
            writer.add_scalar("train/wd", new_wd, global_step)
            writer.add_scalar("time/iter_ms", iter_ms, global_step)
            writer.add_scalar("time/gpu_ms", gpu_ms, global_step)
            writer.add_scalar("time/data_ms", data_elapsed_ms, global_step)
            if vis is not None:
                writer.add_images("recon/target_pred", vis, global_step, dataformats="NCHW")

        if (step % log_freq == 0) or (step == total_steps - 1) or math.isnan(loss) or math.isinf(loss):
            logger.info(
                "[%d/%d] loss: %.4f (avg %.4f) [lr %.2e] [wd %.2e] [mem %.1f MB] [iter %.1f ms] [gpu %.1f ms] [data %.1f ms]"
                % (
                    step + 1,
                    total_steps,
                    loss,
                    loss_meter.avg,
                    new_lr,
                    new_wd,
                    (torch.cuda.max_memory_allocated() / 1024.0**2) if torch.cuda.is_available() else 0.0,
                    iter_ms,
                    gpu_ms,
                    data_elapsed_ms,
                )
            )

        if (step + 1) % save_freq == 0 or (step == total_steps - 1):
            save_checkpoint(step + 1, latest_path)

    if writer is not None:
        writer.close()
