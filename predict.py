import json
import mimetypes
import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

from cog import BasePredictor, Input, Path

from cog_model_helpers import seed as seed_helper
from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
COMFYUI_LORAS_DIR = "ComfyUI/models/loras"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

# Hardcoded model and LoRA URLs - replace with your specific models
# Example model URL: "https://huggingface.co/YOUR_USERNAME/wan-1.3b-model/resolve/main/wan2.1_t2v_1.3B_bf16.safetensors"
HARDCODED_MODEL_URL = "https://huggingface.co/NSFW-API/NSFW_Wan_1.3b/resolve/main/wan_1.3B_e10.safetensors"
HARDCODED_MODEL_FILENAME = "wan2.1_t2v_1.3B_bf16_custom.safetensors"  # Local filename after download

# Example LoRA URL: "https://huggingface.co/fofr/wan2.1-test-loras/resolve/main/wan2.1-1.3b-minecraft-movie.safetensors"
HARDCODED_LORA_URL = "https://huggingface.co/NSFW-API/NSFW_Wan_1.3b_motion_helper/resolve/main/nsfw_wan_1.3b_motion_helper.safetensors"
HARDCODED_LORA_FILENAME = "custom_lora.safetensors"  # Local filename after download

mimetypes.add_type("image/webp", ".webp")
api_json_file = "workflow.json"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


@dataclass
class Inputs:
    prompt = Input(description="Text prompt for video generation")
    negative_prompt = Input(
        description="Things you do not want to see in your video", default=""
    )
    aspect_ratio = Input(
        description="The aspect ratio of the video. 16:9, 9:16, 1:1, etc.",
        choices=["16:9", "9:16", "1:1"],
        default="16:9",
    )
    frames = Input(
        description="The number of frames to generate (1 to 5 seconds)",
        choices=[17, 33, 49, 65, 81],
        default=81,
    )
    lora_strength_model = Input(
        description="Strength of the LoRA applied to the model. 0.0 is no LoRA.",
        default=1.0,
    )
    lora_strength_clip = Input(
        description="Strength of the LoRA applied to the CLIP model. 0.0 is no LoRA.",
        default=1.0,
    )
    sample_shift = Input(
        description="Sample shift factor", default=8.0, ge=0.0, le=10.0
    )
    sample_guide_scale = Input(
        description="Higher guide scale makes prompt adherence better, but can reduce variation",
        default=5.0,
        ge=0.0,
        le=10.0,
    )
    sample_steps = Input(
        description="Number of generation steps. Fewer steps means faster generation, at the expensive of output quality. 30 steps is sufficient for most prompts",
        default=30,
        ge=1,
        le=60,
    )
    seed = Input(default=seed_helper.predict_seed())
    fast_mode = Input(
        description="Speed up generation with different levels of acceleration. Faster modes may degrade quality somewhat. The speedup is dependent on the content, so different videos may see different speedups.",
        choices=["Off", "Balanced", "Fast"],
        default="Balanced",
    )
    resolution = Input(
        description="The resolution of the video.",
        choices=["480p"],
        default="480p",
    )


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        os.makedirs("ComfyUI/models/loras", exist_ok=True)
        os.makedirs("ComfyUI/models/diffusion_models", exist_ok=True)

        # Download custom model from HuggingFace if it doesn't exist
        model_path = os.path.join("ComfyUI/models/diffusion_models", HARDCODED_MODEL_FILENAME)
        if not os.path.exists(model_path):
            print(f"Downloading custom model from {HARDCODED_MODEL_URL}")
            import subprocess
            subprocess.run([
                "pget", "-f",
                HARDCODED_MODEL_URL,
                model_path
            ], check=True)
            print(f"Model downloaded to {model_path}")

        # Download custom LoRA from HuggingFace if it doesn't exist
        lora_path = os.path.join("ComfyUI/models/loras", HARDCODED_LORA_FILENAME)
        if not os.path.exists(lora_path):
            print(f"Downloading custom LoRA from {HARDCODED_LORA_URL}")
            import subprocess
            subprocess.run([
                "pget", "-f",
                HARDCODED_LORA_URL,
                lora_path
            ], check=True)
            print(f"LoRA downloaded to {lora_path}")

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())
        self.comfyUI.handle_weights(
            workflow,
            weights_to_download=[
                "wan_2.1_vae.safetensors",
                "umt5_xxl_fp16.safetensors",
                "clip_vision_h.safetensors",
            ],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
            self,
            input_file: Path,
            filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    def get_width_and_height(self, resolution: str, aspect_ratio: str):
        sizes = {
            "480p": {
                "16:9": (832, 480),
                "9:16": (480, 832),
                "1:1": (644, 644),
            },
        }
        return sizes[resolution][aspect_ratio]

    def update_workflow(self, workflow, **kwargs):
        # Don't set the model filename here - it will be set after load_workflow
        # Just ensure the node exists
        if "37" not in workflow:
            workflow["37"] = {
                "inputs": {
                    "unet_name": "",
                    "weight_dtype": "default"
                },
                "class_type": "UNETLoader",
                "_meta": {
                    "title": "Load Diffusion Model"
                }
            }

        positive_prompt = workflow["6"]["inputs"]
        positive_prompt["text"] = kwargs["prompt"]

        negative_prompt = workflow["7"]["inputs"]
        negative_prompt["text"] = f"nsfw, {kwargs['negative_prompt']}"

        sampler = workflow["3"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["cfg"] = kwargs["sample_guide_scale"]
        sampler["steps"] = kwargs["sample_steps"]

        shift = workflow["48"]["inputs"]
        shift["shift"] = kwargs["sample_shift"]

        # Remove image-to-video nodes (1.3b doesn't support it)
        nodes_to_delete = ["40", "55", "56", "57", "58", "59", "60"]
        for node_id in nodes_to_delete:
            if node_id in workflow:
                del workflow[node_id]

        # Set up text-to-video generation
        if "40" not in workflow:
            # Add empty latent video node
            workflow["40"] = {
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "length": kwargs["frames"],
                    "batch_size": 1
                },
                "class_type": "EmptyHunyuanLatentVideo",
                "_meta": {
                    "title": "EmptyHunyuanLatentVideo"
                }
            }

        width, height = self.get_width_and_height(
            kwargs["resolution"], kwargs["aspect_ratio"]
        )
        empty_latent_video = workflow["40"]["inputs"]
        empty_latent_video["length"] = kwargs["frames"]
        empty_latent_video["width"] = width
        empty_latent_video["height"] = height

        sampler["model"] = ["48", 0]
        sampler["positive"] = ["6", 0]
        sampler["negative"] = ["7", 0]
        sampler["latent_image"] = ["40", 0]

        # Tea cache settings for 1.3b
        thresholds = {
            "1.3b": {
                "Balanced": 0.07,
                "Fast": 0.08,
                "coefficients": "1.3B",
            },
        }

        fast_mode = kwargs["fast_mode"]
        if fast_mode == "Off":
            # Turn off tea cache
            if "54" in workflow:
                del workflow["54"]
            workflow["49"]["inputs"]["model"] = ["37", 0]
        else:
            if "54" not in workflow:
                workflow["54"] = {
                    "inputs": {
                        "rel_l1_thresh": 0.07,
                        "start_percent": 0.1,
                        "end_percent": 1,
                        "cache_device": "offload_device",
                        "coefficients": "1.3B",
                        "model": ["37", 0]
                    },
                    "class_type": "WanVideoTeaCacheKJ",
                    "_meta": {
                        "title": "WanVideo Tea Cache (native)"
                    }
                }
            tea_cache = workflow["54"]["inputs"]
            tea_cache["coefficients"] = thresholds["1.3b"]["coefficients"]
            tea_cache["rel_l1_thresh"] = thresholds["1.3b"][fast_mode]

        # Always use the hardcoded LoRA with local filename
        if "49" not in workflow:
            workflow["49"] = {
                "inputs": {
                    "lora_name": HARDCODED_LORA_FILENAME,
                    "strength_model": kwargs["lora_strength_model"],
                    "strength_clip": kwargs["lora_strength_clip"],
                    "model": ["54", 0] if fast_mode != "Off" else ["37", 0],
                    "clip": ["38", 0]
                },
                "class_type": "LoraLoader",
                "_meta": {
                    "title": "Load LoRA"
                }
            }
        else:
            lora_loader = workflow["49"]["inputs"]
            lora_loader["lora_name"] = HARDCODED_LORA_FILENAME
            lora_loader["strength_model"] = kwargs["lora_strength_model"]
            lora_loader["strength_clip"] = kwargs["lora_strength_clip"]

    def generate(
            self,
            prompt: str,
            negative_prompt: Optional[str] = None,
            aspect_ratio: str = "16:9",
            frames: int = 81,
            lora_strength_model: float = 1.0,
            lora_strength_clip: float = 1.0,
            fast_mode: str = "Balanced",
            sample_shift: float = 8.0,
            sample_guide_scale: float = 5.0,
            sample_steps: int = 30,
            seed: Optional[int] = None,
            resolution: str = "480p",
    ) -> List[Path]:
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        # 1.3b only supports text-to-video and 480p
        if resolution != "480p":
            print("Warning: 1.3b only supports 480p resolution")
            resolution = "480p"

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        # Update workflow settings (but not the model filename yet)
        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            frames=frames,
            aspect_ratio=aspect_ratio,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            resolution=resolution,
        )

        # Load workflow (with empty model name to avoid validation error)
        wf = self.comfyUI.load_workflow(workflow)

        # Now set the custom model filename after load_workflow
        wf["37"]["inputs"]["unet_name"] = HARDCODED_MODEL_FILENAME

        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        return self.comfyUI.get_files(OUTPUT_DIR, file_extensions=["mp4"])


class StandaloneLoraPredictor(Predictor):
    def predict(
            self,
            prompt: str = Inputs.prompt,
            negative_prompt: str = Inputs.negative_prompt,
            aspect_ratio: str = Inputs.aspect_ratio,
            frames: int = Inputs.frames,
            resolution: str = Inputs.resolution,
            lora_strength_model: float = Inputs.lora_strength_model,
            lora_strength_clip: float = Inputs.lora_strength_clip,
            fast_mode: str = Inputs.fast_mode,
            sample_steps: int = Inputs.sample_steps,
            sample_guide_scale: float = Inputs.sample_guide_scale,
            sample_shift: float = Inputs.sample_shift,
            seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            frames=frames,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            seed=seed,
            resolution=resolution,
        )


class TrainedLoraPredictor(Predictor):
    def predict(
            self,
            prompt: str = Inputs.prompt,
            negative_prompt: str = Inputs.negative_prompt,
            aspect_ratio: str = Inputs.aspect_ratio,
            frames: int = Inputs.frames,
            resolution: str = Inputs.resolution,
            lora_strength_model: float = Inputs.lora_strength_model,
            lora_strength_clip: float = Inputs.lora_strength_clip,
            fast_mode: str = Inputs.fast_mode,
            sample_steps: int = Inputs.sample_steps,
            sample_guide_scale: float = Inputs.sample_guide_scale,
            sample_shift: float = Inputs.sample_shift,
            seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        return self.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            frames=frames,
            lora_strength_model=lora_strength_model,
            lora_strength_clip=lora_strength_clip,
            fast_mode=fast_mode,
            sample_shift=sample_shift,
            sample_guide_scale=sample_guide_scale,
            sample_steps=sample_steps,
            seed=seed,
            resolution=resolution,
        )
