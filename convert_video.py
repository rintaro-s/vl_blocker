import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
from moviepy.editor import VideoFileClip, AudioFileClip

# Use backend converter directly
from backend.voice_converter import init_converter, converter, CONFIG


def resample_mono(data: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr == target_sr:
        return data.astype(np.float32)
    # simple linear resample
    n_new = int(len(data) * target_sr / sr)
    if n_new <= 0:
        return np.zeros(0, dtype=np.float32)
    return np.interp(np.linspace(0, len(data) - 1, n_new), np.arange(len(data)), data).astype(np.float32)


def chunk_wav_bytes(y: np.ndarray, sr: int) -> bytes:
    import io
    buf = io.BytesIO()
    sf.write(buf, y, sr, subtype='PCM_16', format='WAV')
    buf.seek(0)
    return buf.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='yukkuri.mp4', help='input video file')
    parser.add_argument('--output', default='yukkuri_converted.mp4', help='output video file')
    parser.add_argument('--mode', default='mute', choices=['mute', 'rvc', 'dsp'], help='mute: remove voice, rvc: use model if available, dsp: pitch/filters')
    parser.add_argument('--target', default='zundamon-1', help='target RVC model name when mode=rvc; or dsp target speaker')
    parser.add_argument('--keep-wav', action='store_true', help='also write wav sidecar')
    args = parser.parse_args()

    init_converter()
    from backend.voice_converter import converter as global_converter
    conv = global_converter
    if conv is None:
        raise RuntimeError("converter not initialized (global)")

    src_path = Path(args.input)
    if not src_path.exists():
        raise FileNotFoundError(f"input not found: {src_path}")

    clip = VideoFileClip(str(src_path))
    if clip.audio is None:
        raise RuntimeError('video has no audio')
    # write audio to array
    audio_fps = int(clip.audio.fps)
    audio = clip.audio.to_soundarray(fps=audio_fps)  # shape (n, channels)
    audio_np = resample_mono(audio, audio_fps, CONFIG['SAMPLE_RATE'])

    # chunk into 1s segments
    chunk_size = CONFIG['CHUNK_SIZE']
    out_segments = []

    i = 0
    while i < len(audio_np):
        seg = audio_np[i:i+chunk_size]
        if seg.size < chunk_size:
            pad = np.zeros(chunk_size, dtype=np.float32)
            pad[:seg.size] = seg
            seg = pad
        wav_bytes = chunk_wav_bytes(seg, CONFIG['SAMPLE_RATE'])
        # decide target param
        if args.mode == 'mute':
            target = 'mute'
        elif args.mode == 'rvc':
            target = args.target
        else:
            target = args.target  # dsp target speaker id
        wav_out, meta = conv.process_chunk(wav_bytes, target_speaker=target)
        # read wav_out back to float
        import io
        data, sr = sf.read(io.BytesIO(wav_out), dtype='float32')
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        out_segments.append(data.astype(np.float32))
        i += chunk_size

    out_np = np.concatenate(out_segments) if out_segments else np.zeros(0, dtype=np.float32)

    # build audio clip
    tmp_wav = src_path.with_suffix('.converted.wav')
    sf.write(str(tmp_wav), out_np, CONFIG['SAMPLE_RATE'], subtype='PCM_16')

    if args.keep_wav:
        print(f"Wrote: {tmp_wav}")

    new_audio = AudioFileClip(str(tmp_wav))
    new_clip = clip.set_audio(new_audio)
    new_clip.write_videofile(str(Path(args.output)), audio_codec='aac', codec='libx264')


if __name__ == '__main__':
    main()
