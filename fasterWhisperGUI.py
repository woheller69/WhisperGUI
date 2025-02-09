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
from faster_whisper import WhisperModel



class WhisperGUI:
    def __init__(self):
        self.output_window = None
        self.recording_thread = None
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
        
        start_button = tk.Button(self.root, text="Press & Hold while speaking")
        start_button.pack()

        # Bind the button press and release events. <Button-1> is the event for a mouse button press. 1 refers to the left mouse button.
        start_button.bind("<Button-1>", self.start)
        start_button.bind("<ButtonRelease-1>", self.generate)
        
        
        exit_button = tk.Button(self.root, text="Exit", command=self.root.destroy)

        start_button.pack(side='left', padx=(20, 0))
        exit_button.pack(side='right', padx=(0, 20))
    
        self.new_whisper_session()
        
        self.root.mainloop()
        
    def new_whisper_session(self):   
        #Load whisper model
        # Initialize the Whisper Model
        self.model = WhisperModel("small", device="cpu", compute_type="int8")
        

    def init_record(self):
        # Capture audio from the microphone
        self.capture_audio()

    def generate(self, event):
        self.recording = False
        self.recording_thread.join()
            
        # Resample audio to 16,000 Hz
        target_sample_rate = 16000
        if self.sample_rate != target_sample_rate:
            print(f"Resampling audio from {self.sample_rate} Hz to {target_sample_rate} Hz")
            self.audio_data = librosa.resample(self.audio_data.astype(np.float32) / 32768.0, orig_sr=self.sample_rate, target_sr=target_sample_rate)
            self.sample_rate = target_sample_rate

        print(f"Whisper processing")


        # Generate transcription
        try:
            segments, info = self.model.transcribe(self.audio_data)
            for segment in segments:
                self.output_window.insert(tk.END, segment.text)
                self.output_window.yview(tk.END)
        except Exception as e:
            print(f"Error generating transcription: {e}")
            exit(1)

    def start(self, event):
        if self.recording_thread is None:
            self.recording_thread = threading.Thread(target=self.init_record)
            self.recording_thread.start()
    
    def exit(self):
        del self.gpt4all_instance
        quit()

    def on_closing(self):
        del self.gpt4all_instance
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
