import os
import argparse
import torch
from diffusers.utils import export_to_video
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from guidance_pipeline import CustomWanPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Wan T2V 批量视频生成")

    # 必需参数
    parser.add_argument("--prompt_file", type=str, required=True,
                        help="正向 prompts 文件路径 (txt，每行一个 prompt)")
    parser.add_argument("--vio_prompt_file", type=str, required=True,
                        help="违背 prompts 文件路径 (txt，每行一个 vio prompt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出视频保存目录")

    # 可选参数
    parser.add_argument("--model_id", type=str, default="/data/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=8.0)
    parser.add_argument("--vio_scale", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--flow_shift", type=float, default=3.0, help="480p=3.0, 720p=5.0")
    parser.add_argument("--start_num", type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    # ===== 模型加载 =====
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", torch_dtype=torch.float16
    )

    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=args.flow_shift
    )

    pipe = CustomWanPipeline.from_pretrained(
        args.model_id, vae=vae, torch_dtype=torch.bfloat16
    )
    pipe.scheduler = scheduler

    # 内存优化
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()

    # ====== 输出目录 ======
    os.makedirs(args.output_dir, exist_ok=True)

    # ====== 读取 prompts ======
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    with open(args.vio_prompt_file, "r") as f:
        vio_prompts = [line.strip() for line in f if line.strip()]

    num = args.start_num
    # prompts = prompts[6:]
    # vio_prompts = vio_prompts[6:]

    assert len(prompts) == len(vio_prompts), "❌ prompts 和 vio_prompts 数量不一致！"

    # ====== 批量生成 ======
    for i, (prompt, vio_prompt) in enumerate(zip(prompts, vio_prompts)):
        print(f"\n[生成视频 {i+1+num}]")
        print(f" 正常 Prompt: {prompt}")
        print(f" 违背 Prompt: {vio_prompt}")

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = pipe(
                prompt=prompt,
                vio_prompt=vio_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps
            ).frames[0]

        save_path = os.path.join(args.output_dir, f"video_{i+1+num}.mp4")
        export_to_video(output, save_path, fps=args.fps)
        print(f"✅ 已保存: {save_path}")


if __name__ == "__main__":
    main()