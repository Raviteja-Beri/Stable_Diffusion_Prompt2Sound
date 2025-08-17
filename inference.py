# inference.py
from diffusers import StableAudioPipeline
import torch
import soundfile as sf
import uuid
import os

# Load the model (this will take some VRAM)
pipe = StableAudioPipeline.from_pretrained(
    "stabilityai/stable-audio-open-1.0",
    torch_dtype=torch.float16
).to("cuda")

def generate_audio(prompt, negative_prompt="low quality"):
    generator = torch.Generator("cuda").manual_seed(0)
    audio = pipe(
        prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=200,
        audio_end_in_s=10.0,
        num_waveforms_per_prompt=1,
        generator=generator
    ).audios

    output = audio[0].T.float().cpu().numpy()
    os.makedirs("outputs", exist_ok=True)
    file_path = f"outputs/{uuid.uuid4()}.wav"
    sf.write(file_path, output, pipe.vae.sampling_rate)
    return file_path

