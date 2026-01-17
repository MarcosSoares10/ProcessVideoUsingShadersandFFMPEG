import os
import subprocess
import numpy as np
from pathlib import Path
from gpu_filters import GPUProcessor
from esrgan import ESRGAN
import time

start_time = time.time()
# ---------------- Configurações ----------------
project_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_dir, "Source_Videos")
result_base = os.path.join(project_dir, "Results")
os.makedirs(result_base, exist_ok=True)

use_esrgan = False  # True = aplica ESRGAN x2, False = sem upscale

videos = [
    os.path.join(source_dir, f)
    for f in os.listdir(source_dir)
    if f.lower().endswith(".mp4")
]

gpu = GPUProcessor()
esrgan = ESRGAN(use_x2=True) if use_esrgan else None

# -----------------------------------------------------------
# Funções auxiliares
# -----------------------------------------------------------
def video_has_audio(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return b"audio" in result.stdout

def get_video_info(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "csv=p=0",
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    w, h, fps_str = result.stdout.decode().strip().split(",")

    if "/" in fps_str:
        n, d = map(float, fps_str.split("/"))
        fps = n / d
    else:
        fps = float(fps_str)

    return int(w), int(h), fps

def start_ffmpeg_reader(video_path):
    return subprocess.Popen(
        ["ffmpeg", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, bufsize=10**8
    )

def start_ffmpeg_writer(video_path, w, h, fps, codec, has_audio, original_video_path):
    cmd = ["ffmpeg", "-y"]

    if has_audio:
        cmd += ["-i", original_video_path]

    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-"
    ]

    if has_audio:
        cmd += ["-map", "1:v:0", "-map", "0:a:0", "-c:a", "copy"]
    else:
        cmd += ["-map", "0:v:0"]

    cmd += [
        "-c:v", codec,
        "-crf", "20",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        video_path
    ]

    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

# -----------------------------------------------------------
# Pipeline principal
# -----------------------------------------------------------
for video in videos:
    name = Path(video).stem
    print(f"\n Processando vídeo: {name}")

    w, h, fps = get_video_info(video)
    has_audio = video_has_audio(video)

    reader = start_ffmpeg_reader(video)

    # Le o primeiro frame para descobrir a resolução final
    frame_size = w * h * 3
    raw = reader.stdout.read(frame_size)
    if not raw:
        print("Vídeo vazio")
        continue

    frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
    frame = gpu.process(frame)

    if use_esrgan:
        frame = esrgan.upscale(frame)

    h_out, w_out, _ = frame.shape  # Atualiza resolução após ESRGAN se necessário

    # Cria os writers H265 e H264 com a resolução correta
    #out_h265 = os.path.join(result_base, f"{name}_final_h265.mp4")
    out_h264 = os.path.join(result_base, f"{name}_final_h264.mp4")

    #writer_h265 = start_ffmpeg_writer(out_h265, w_out, h_out, fps, codec="libx265", has_audio=has_audio, original_video_path=video)
    writer_h264 = start_ffmpeg_writer(out_h264, w_out, h_out, fps, codec="libx264", has_audio=has_audio, original_video_path=video)

    # Escreve o primeiro frame
    #writer_h265.stdin.write(frame.tobytes())
    writer_h264.stdin.write(frame.tobytes())

    # Continua processando o resto dos frames
    while True:
        raw = reader.stdout.read(frame_size)
        if not raw:
            break

        frame = np.frombuffer(raw, np.uint8).reshape((h, w, 3))
        frame = gpu.process(frame)

        if use_esrgan:
            frame = esrgan.upscale(frame)

        #writer_h265.stdin.write(frame.tobytes())
        writer_h264.stdin.write(frame.tobytes())

    # Fecha FFmpeg
    #writer_h265.stdin.close()
    #writer_h265.wait()

    writer_h264.stdin.close()
    writer_h264.wait()

    reader.stdout.close()
    reader.wait()

    print(f"Vídeo final gerado")

    elapsed = time.time() - start_time  # End timer
    print(f"Tempo de processamento: {elapsed:.2f} segundos")
