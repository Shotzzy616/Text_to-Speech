from TTS.api import TTS
import simpleaudio as sa
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda")

# generate speech by cloning a voice using default settings
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent. That said, Hello Claude How may i assist you?",
                file_path="output.wav",
                speaker_wav="X:/Friday/Voices/en_sample.wav",
                language="en")

# Play the generated audio file
wave_obj = sa.WaveObject.from_wave_file("output.wav")
play_obj = wave_obj.play()
play_obj.wait_done()