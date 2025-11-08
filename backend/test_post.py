"""
Test streaming: read a WAV file, send it as chunks to backend /convert endpoint,
receive converted chunks back, combine them into output.

Usage:
  python test_post.py [path\to\input.wav]

If no path given, uses 'test.wav' in current directory.
Output is saved to 'out_from_test_post.wav'.

This simulates what the browser extension will do:
1. Read audio file (or capture chunks in real browser)
2. Send chunks of ~4096 samples to /convert
3. Receive converted chunks back
4. Combine into output file
"""
import sys
import os
import io
import struct
import numpy as np
import soundfile as sf
import requests

BACKEND = os.environ.get('YV_BACKEND', 'http://127.0.0.1:5000')
CHUNK_SAMPLES = 4096  # Chunk size in samples (for 16kHz, ~0.25 seconds)
SAMPLE_RATE = 16000

def read_wav(path):
    """Read WAV file and return (samples_float32, sample_rate)"""
    try:
        data, sr = sf.read(path, dtype='float32')
        print(f"[OK] Loaded {path}: shape={data.shape}, sr={sr}")
        return data, sr
    except Exception as e:
        print(f"[ERR] failed to read {path}: {e}")
        sys.exit(2)

def float32_to_bytes(data):
    """Convert float32 numpy array to bytes"""
    return data.astype(np.float32).tobytes()

def bytes_to_float32(data):
    """Convert bytes to float32 numpy array
    Backend returns int16 PCM, so convert from int16 to float32
    """
    try:
        # Try int16 first (backend returns int16)
        samples_int16 = np.frombuffer(data, dtype=np.int16)
        return samples_int16.astype(np.float32) / 32768.0
    except Exception:
        # Fallback to float32
        return np.frombuffer(data, dtype=np.float32)

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else 'test.wav'
    if not os.path.exists(path):
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    # Read input WAV
    audio_data, sr = read_wav(path)
    
    # Convert stereo to mono if needed
    if audio_data.ndim > 1:
        print(f"[*] Converting stereo to mono...")
        audio_data = np.mean(audio_data, axis=1)
        print(f"[OK] Mono: shape={audio_data.shape}")
    
    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        print(f"[*] Resampling from {sr}Hz to {SAMPLE_RATE}Hz...")
        n_new = int(len(audio_data) * SAMPLE_RATE / sr)
        audio_data = np.interp(
            np.linspace(0, len(audio_data) - 1, n_new),
            np.arange(len(audio_data)),
            audio_data
        )
        sr = SAMPLE_RATE
        print(f"[OK] Resampled: new shape={audio_data.shape}")
    
    # Ensure float32
    audio_data = audio_data.astype(np.float32)
    
    print(f"\n[*] Starting streaming chunk test:")
    print(f"    Total samples: {len(audio_data)}")
    print(f"    Chunk size: {CHUNK_SAMPLES} samples")
    print(f"    Expected chunks: {(len(audio_data) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES}")
    
    url = BACKEND.rstrip('/') + '/convert'
    converted_chunks = []
    total_sent = 0
    total_received = 0
    
    # Stream chunks
    for i in range(0, len(audio_data), CHUNK_SAMPLES):
        chunk = audio_data[i:i + CHUNK_SAMPLES]
        chunk_bytes = float32_to_bytes(chunk)
        
        try:
            print(f"    [{i:06d}] Sending {len(chunk)} samples ({len(chunk_bytes)} bytes)...", end='', flush=True)
            r = requests.post(
                url,
                data=chunk_bytes,
                headers={'Content-Type': 'application/octet-stream'},
                timeout=30
            )
            print(f" status={r.status_code}", end='', flush=True)
            
            if r.status_code != 200:
                print(f"\n[ERR] Error at chunk {i}: HTTP {r.status_code}")
                try:
                    print(f"      Response: {r.json()}")
                except:
                    print(f"      Response: {r.text[:200]}")
                sys.exit(3)
            
            # Parse response as int16 PCM
            converted_chunk = bytes_to_float32(r.content)
            converted_chunks.append(converted_chunk)
            total_sent += len(chunk)
            total_received += len(converted_chunk)
            print(f" <- {len(converted_chunk)} samples")
            
        except Exception as e:
            print(f"\n[ERR] Error at chunk {i}: {e}")
            sys.exit(4)
    
    # Combine all chunks
    print(f"\n[OK] All chunks processed")
    print(f"     Total sent: {total_sent} samples")
    print(f"     Total received: {total_received} samples")
    
    combined = np.concatenate(converted_chunks) if converted_chunks else np.array([], dtype=np.float32)
    print(f"     Combined output: {len(combined)} samples")
    
    # Save output WAV
    out_path = 'out_from_test_post.wav'
    try:
        sf.write(out_path, combined, samplerate=SAMPLE_RATE, subtype='FLOAT')
        out_size = os.path.getsize(out_path)
        print(f"\n[OK] Saved {out_path} ({out_size} bytes)")
    except Exception as e:
        print(f"\n[ERR] Failed to save {out_path}: {e}")
        sys.exit(5)

if __name__ == '__main__':
    main()
