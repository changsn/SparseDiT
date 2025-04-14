# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import os


def main(args):
    # Setup PyTorch:
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        num_poolingformer=args.num_poolingformer,
        num_dumbbell=args.num_dumbbell,
        num_normal_block=args.num_normal_block,
        semantic_size=args.semantic_size,
        dumbbell_group=args.dumbbell_group,
        attn_relat_pos=args.attn_relat_pos,
        temp=args.temp,
        use_checkpoint=False
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    # ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    # state_dict = find_model(ckpt_path)
    # model.load_state_dict(state_dict)
    # model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"/mnt/workspace/diffusion/fast-DiT/stabilityai/sd-vae-ft-{args.vae}").to(device)
    if args.ckpt.endswith('.pt'):
        ckpt_paths = [args.ckpt]
        output_path = os.path.join('/', *(args.ckpt.split('/')[:-2] + ['sample']))
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)
    else:
        ckpt_paths = [os.path.join(args.ckpt, i) for i in os.listdir(args.ckpt)]
        output_path = os.path.join(args.ckpt, 'sample')
        if os.path.exists(output_path):
            pass
        else:
            os.makedirs(output_path)
    # Labels to condition the model with (feel free to change):
    # class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    # class_labels = [1, 2, 25, 31, 679, 803, 820, 934]
    class_labels = [388 for i in range(8)]

    # Create sampling noise:
    n = len(class_labels)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    for ckpt_path in ckpt_paths:
        torch.manual_seed(args.seed)
        z = torch.randn(n, 4, latent_size, latent_size, device=device)
        z = torch.cat([z, z], 0)
        state_dict = find_model(ckpt_path)
        model.load_state_dict(state_dict)
        model.eval()  # important!
        # Sample images:
        if args.mode == 'ddpm':
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device, vae=None
            )
        elif args.mode == 'ddim':
            samples = diffusion.ddim_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
            )
        else:
            raise Exception()
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        save_image(samples, os.path.join(output_path, ckpt_path.split('/')[-1][:-2]+'png'), nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    parser.add_argument("--mode", type=str, choices=['ddpm', 'ddim'], default='ddpm')
    parser.add_argument("--num_poolingformer", type=int, default=2)
    parser.add_argument("--num_dumbbell", type=int, default=4)
    parser.add_argument("--num_normal_block", type=int, default=2)
    parser.add_argument("--semantic_size", type=int, default=16)
    parser.add_argument("--dumbbell_group", type=int, nargs='+', default=[1, 3, 1, 1])
    parser.add_argument("--attn_relat_pos", type=str, choices=['sin', 'relat'], default=None)
    parser.add_argument("--temp", type=float, default=1.)
    parser.add_argument("--checkpoint_mode", type=str, default='ema')
    args = parser.parse_args()
    main(args)
