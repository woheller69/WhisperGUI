#!/usr/bin/env python3

import os
import io
import sys
import typer
import tkinter as tk
from tkinter import scrolledtext
from typing_extensions import Annotated
import threading
import time
import pyaudio
import numpy as np
import librosa
import tensorflow as tf
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration



class WhisperGUI:
    def __init__(self):
        self.output_window = None
        self.recording_thread = None
        self.processor = None
        self.model = None
        self.audio_data = None
        self.sample_rate = None
        self.recording = False
    
    def run(self):
        self.root = tk.Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.title('WHISPER GUI')
        self.root.geometry("768x768")

        self.output_window = scrolledtext.ScrolledText(self.root, wrap=tk.WORD)
        self.output_window.pack(side='top', fill='both', expand=True)

        generate_button = tk.Button(self.root, text="Stop Recording and Transcribe", command=self.generate)
        start_button = tk.Button(self.root, text="Start Recording", command=self.start)
        exit_button = tk.Button(self.root, text="Exit", command=self.root.destroy)

        start_button.pack(side='left', padx=(20, 0))
        generate_button.pack(side='left', padx=(20, 0))
        exit_button.pack(side='right', padx=(0, 20))
    
        self.new_whisper_session()
        
        self.root.mainloop()
        
    def new_whisper_session(self):   
        #Load whisper model
        # Initialize the Whisper Processor and Model
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.model = TFWhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
        

    def init_record(self):
        # Capture audio from the microphone
        self.capture_audio()

    def generate(self):
        self.recording = False
        self.recording_thread.join()
            
        # Resample audio to 16,000 Hz
        target_sample_rate = 16000
        if self.sample_rate != target_sample_rate:
            print(f"Resampling audio from {self.sample_rate} Hz to {target_sample_rate} Hz")
            self.audio_data = librosa.resample(self.audio_data.astype(np.float32) / 32768.0, orig_sr=self.sample_rate, target_sr=target_sample_rate)
            self.sample_rate = target_sample_rate

        # Preprocess the audio data
        try:
            print(f"Whisper processing")
            input_features = self.processor(self.audio_data, sampling_rate=self.sample_rate, return_tensors="tf").input_features
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            exit(1)

        # Generate transcription
        try:
            forced_decoder_ids=[[2, 50359], [3, 50363]]
            generated_ids = self.model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            self.output_window.insert(tk.END, transcription[0])
            self.output_window.yview(tk.END)
        except Exception as e:
            print(f"Error generating transcription: {e}")
            exit(1)

    def start(self):
        if self.recording_thread is None:
            self.recording_thread = threading.Thread(target=self.init_record)
            self.recording_thread.start()
    
    def exit(self):
        quit()

    def on_closing(self):
        self.root.destroy()
        
    # Function to capture audio from the microphone
    def capture_audio(self, duration=5, sample_rate=44100, chunk_size=2048, channels=1, audio_format=pyaudio.paInt16, device_index=0):
        audio = pyaudio.PyAudio()
        print("Recording started")
        self.sample_rate = sample_rate
    
        try:
            stream = audio.open(format=audio_format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size, input_device_index=0)
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            audio.terminate()
            return
    
        frames = []
        self.recording = True
    
        try:
            while self.recording:
                data = stream.read(chunk_size, exception_on_overflow=False)
                frames.append(data)
        except Exception as e:
            print(f"Error reading audio data: {e}")
            self.recording = False
       
        stream.stop_stream()
        stream.close()
        audio.terminate()
    
        if len(frames) == 0:
            print("No audio data captured.")
            return
    
        self.audio_data = np.frombuffer(b''.join(frames), dtype=np.int16)
        print("Recording finished")
        self.recording_thread = None

if __name__ == "__main__":
    gui = WhisperGUI()
    typer.run(gui.run)

