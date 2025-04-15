import os
import re
import numpy as np
import soundfile as sf

def combine_audio_files(output_dir: str, combined_filename: str = "podcast.wav"):
    """
    Combine all WAV files in the output directory into a single podcast file.

    Args:
        output_dir: Directory containing individual audio files
        combined_filename: Filename for the combined podcast
    """
    audio_files = [
        f
        for f in os.listdir(output_dir)
        if f.endswith(".wav") and f != combined_filename
    ]
    def get_file_number(filename):
        match = re.search(r"output_(\d+)\.wav", filename)
        if match:
            return int(match.group(1))
        return 0
    audio_files.sort(key=get_file_number)
    print(f"Audio files in order: {audio_files}")
    if not audio_files:
        print("No audio files found to combine")
        return
    print(f"Combining {len(audio_files)} audio files...")
    audio_segments = []
    samplerate = None
    for audio_file in audio_files:
        file_path = os.path.join(output_dir, audio_file)
        data, file_samplerate = sf.read(file_path)
        audio_segments.append(data)
        if samplerate is None:
            samplerate = file_samplerate
    pause_duration = int(0.2 * samplerate)
    pause = np.zeros(pause_duration)
    combined = np.array([])
    for segment in audio_segments:
        combined = np.append(combined, segment)
        combined = np.append(combined, pause)
    combined_path = os.path.join(output_dir, combined_filename)
    sf.write(combined_path, combined, samplerate)
    print(f"Combined podcast saved to {combined_path}")
