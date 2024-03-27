import argparse
import torch
import re
import time
import gradio as gr
from moondream import detect_device, LATEST_REVISION
from threading import Thread
from transformers import TextIteratorStreamer, AutoTokenizer, AutoModelForCausalLM
from moviepy.editor import VideoFileClip
from PIL import Image
import io
import numpy as np
import tempfile


parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

if args.cpu:
    device = torch.device("cpu")
    dtype = torch.float32
else:
    device, dtype = detect_device()
    if device != torch.device("cpu"):
        print("Using device:", device)
        print("If you run into issues, pass the `--cpu` flag to this script.")
        print()

model_id = "vikhyatk/moondream2"
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
moondream = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=LATEST_REVISION
).to(device=device, dtype=dtype)
moondream.eval()

def extract_frames(video_bytes, num_frames=10):
    """Extracts a set of frames evenly spaced throughout the video."""
    clip = VideoFileClip(io.BytesIO(video_bytes))
    duration = clip.duration
    frames = []
    for i in range(num_frames):
        t = (duration / num_frames) * i
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        frames.append(img)
    clip.close()
    return frames

def summarize_descriptions(descriptions):
    """Summarizes a list of descriptions into a concise paragraph."""
    # Placeholder for summarization logic
    summary = " ".join(descriptions)  # Simple concatenation for demonstration
    if len(summary) > 500:
        return summary[:497] + "..."
    return summary


def answer_question(img, prompt="explain in detail what is happening in this image which represents a frame of video while keeping in mind the frames which came before so that this explination will be best formatted to be used in conjunction with others like it for a final video summary to be generated from all of the image summaries created here."):
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        clean_text = re.sub("<$|END$", "", new_text)
        buffer += clean_text
        yield buffer.strip("<END")

def analyze_video(video):
    if video is None:
        return "No video uploaded."
    else:
        print("Video data type:", type(video))  # Debugging statement
        description = process_video(video)
        output.update(description)

def process_video(video):
    # Ensure video content is not None and is in bytes
    if video is None or not isinstance(video, bytes):
        raise ValueError("Invalid video content. Please upload a valid video file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video)
        temp_video_file.flush()  # Ensure all data is written to disk
        temp_video_path = temp_video_file.name

    try:
        frames = extract_frames(temp_video_path)  # Now passing the file path instead of bytes
        descriptions = [next(answer_question(frame)) for frame in frames]
        summary = summarize_descriptions(descriptions)
    finally:
        os.remove(temp_video_path)  # Clean up the temporary file

    return summary


with gr.Blocks() as demo:
    gr.Markdown("# 🌔 Moondream Video Analysis")
    video = gr.File(label="Upload a Video", type="binary")
    output = gr.Textbox(label="Video Description")

    @video.change()
    def analyze_video(video):
        description = process_video(video)
        output.update(description)

demo.launch(debug=True)
