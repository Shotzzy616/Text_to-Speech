import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import time

# Function to capture audio from the microphone
def record_audio(duration, fs):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype=np.int16)
    sd.wait()  # Wait until recording is finished
    print("Recording complete.")
    return audio

# Transcribe the captured audio
def transcribe_audio(audio, fs, model):
    # Save the audio to a temporary file
    temp_wav_file = 'temp_audio.wav'
    write(temp_wav_file, fs, audio)
    
    # Transcribe the audio file
    segments, info = model.transcribe(temp_wav_file, beam_size=5)
    
    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
    
    transcribed_text = ""
    
    for segment in segments:
        print(transcribed_text.strip())
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    
    return segments

if __name__ == "__main__":
    # Model configuration
    model_size = "large-v3"
    model = WhisperModel(model_size,  device="cuda", compute_type="float16")
    
    # Recording configuration
    duration = 5  # seconds
    fs = 16000  # Sample rate
    
    while True:
        
        # Capture audio from the microphone
        audio = record_audio(duration, fs)
        
        # Transcribe the captured audio
        segments = transcribe_audio(audio, fs, model)
