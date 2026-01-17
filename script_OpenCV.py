# py -3.11 -m venv venv
# venv\Scripts\activate
# python -m pip install --upgrade pip setuptools wheel
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# pip install opencv-python PyOpenGL glfw numpy tqdm realesrgan basicsr
import os, cv2, subprocess
import time
from tqdm import tqdm
from pathlib import Path
from gpu_filters import GPUProcessor
from esrgan import ESRGAN
import shutil
import numpy as np


# -----------------------------------------------------------
# NOVO: detectar se o vídeo possui áudio
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

# -----------------------------------------------------------
# EXTRAÇÃO DE FRAMES + ÁUDIO
# -----------------------------------------------------------
def extract_frames(video_path,out):
    os.makedirs(out,exist_ok=True)
    TRIM_START = 0
    TRIM_END = 0

    # Pega duração do vídeo
    duration_cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    result = subprocess.run(duration_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    total_duration = float(result.stdout.decode().strip())
    trimmed_duration = total_duration - TRIM_START - TRIM_END

    # Detecta FPS
    fps_cmd = [
        "ffprobe", "-v", "0", "-of", "csv=p=0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        video_path
    ]
    fps_result = subprocess.run(fps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fps_str = fps_result.stdout.decode().strip()
    if '/' in fps_str:
        num, denom = map(float, fps_str.split('/'))
        fps = num / denom if denom != 0 else 0
    else:
        fps = float(fps_str)

    # extrai frames
    subprocess.run([
        'ffmpeg', '-y', '-i', video_path,
        '-ss', str(TRIM_START), '-t', str(trimmed_duration),
        '-q:v', '1', f'{extracted_dir}/IMG%04d.png'
    ], check=True)

    # verifica áudio
    if video_has_audio(video_path):
        audio_path = os.path.join(extracted_dir, "audio.aac")
        subprocess.run([
            'ffmpeg', '-y', '-i', video_path,
            '-ss', str(TRIM_START), '-t', str(trimmed_duration),
            '-vn', '-acodec', 'copy', audio_path
        ], check=True)
    else:
        print("⚠️ Vídeo sem áudio — pulando extração de áudio.")

    return fps

# -----------------------------------------------------------
# SALVA AS IMAGENS PROCESSADAS EM UM VÍDEO
# -----------------------------------------------------------
def frames_to_video(processed_dir, extracted_dir, result_dir, fps):
    audio_path = os.path.join(extracted_dir, "audio.aac")
    has_audio = os.path.exists(audio_path)

    # vídeo temporário de imagens
    subprocess.run([
        'ffmpeg', '-y', '-framerate', str(fps),
        '-i', f'{processed_dir}/IMG%04d.png',
        '-c:v', 'copy', f'{result_dir}/temp_video.mkv'
    ], check=True)

    # vídeo final com ou sem áudio
    if has_audio:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', f'{result_dir}/temp_video.mkv',
            '-i', audio_path,
            '-c:v', 'copy', '-c:a', 'copy',
            '-map', '0:v:0', '-map', '1:a:0',
            f'{result_dir}/Video_with_audio.mkv'
        ], check=True)
    else:
        print("➡️ Gerando vídeo final sem áudio.")
        os.rename(
            f"{result_dir}/temp_video.mkv",
            f"{result_dir}/Video_with_audio.mkv"
        )

# -----------------------------------------------------------
# CONVERSÃO FINAL H265 / H264
# -----------------------------------------------------------
def SaveVideoToH265(result_dir, video_name, fps, use_scale=True):
    cmd = [
        'ffmpeg', '-y', '-i', f'{result_dir}/Video_with_audio.mkv'
    ]
    if use_scale:
        cmd += ['-vf', "scale='-2:min(1080,ih)'"]
    cmd += ['-r', str(fps),
        '-c:v', 'libx265', '-crf', '17', '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        f'{result_dir}/{video_name}']
    subprocess.run(cmd, check=True)

def SaveVideoToH264(result_dir, video_name, fps, use_scale=True):
    cmd = [
        'ffmpeg', '-y', '-i', f'{result_dir}/Video_with_audio.mkv'
    ]
    if use_scale:
        cmd += ['-vf', "scale='-2:min(1080,ih)'"]
    cmd += ['-r', str(fps),
        '-c:v', 'libx264', '-crf', '17', '-preset', 'slow',
        '-pix_fmt', 'yuv420p',
        f'{result_dir}/{video_name}']
    subprocess.run(cmd, check=True)

def clean_results_directory(results_dir, final_video_names):
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)

        # mantém apenas os vídeos finais
        if item in final_video_names:
            continue

        # remove diretórios e arquivos temporários
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

def CropImageOnVerticalOrientation(Img):
    h, w = Img.shape[:2]
    target_ratio = 9 / 16
    if h / w > (16 / 9):
        new_w = w
        new_h = int(w * (16 / 9))
    else:
        new_h = h
        new_w = int(h * target_ratio)
    x_start = (w - new_w) // 2
    y_start = (h - new_h) // 2
    return Img[y_start:y_start + new_h, x_start:x_start + new_w]

project_dir = os.path.dirname(os.path.abspath(__file__))
source_dir = os.path.join(project_dir, "Source_Videos")
result_base = os.path.join(project_dir, "Results")

os.makedirs(result_base,exist_ok=True)

videos = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.lower().endswith(".mp4")
    ]

gpu=GPUProcessor()
esrgan=ESRGAN(use_x2= True)

start_time = time.time()
final_video_names = []
for video in videos:
    name=Path(video).stem
    extracted_dir=f"{result_base}/{name}_frames"
    processed_dir=f"{result_base}/{name}_processed"
    os.makedirs(processed_dir,exist_ok=True)
    fps = extract_frames(video,extracted_dir)
    print(f"Detected FPS: {fps}")
    frame_files = os.listdir(extracted_dir)
    for i, f in enumerate(tqdm(frame_files, desc=f"Processing {name}")):
        img = cv2.imread(f"{extracted_dir}/{f}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        #img = esrgan.upscale(img,w)
        img = gpu.process(img)
        #img = CropImageOnVerticalOrientation(img)    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{processed_dir}/{f}", img)
    
    frames_to_video(processed_dir, extracted_dir, result_base ,fps)
    SaveVideoToH265(result_base, f"{name}_final_h265.mp4", fps, use_scale=True)
    SaveVideoToH264(result_base, f"{name}_final_h264.mp4", fps, use_scale=True)
    final_video_names.append(f"{name}_final_h265.mp4")
    final_video_names.append(f"{name}_final_h264.mp4")
    
end_time = time.time()

print(f"Tempo total de execução: {end_time - start_time:.2f} segundos")
#clean_results_directory(result_base, final_video_names)