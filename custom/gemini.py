
# from google import genai

# client = genai.Client(api_key="AIzaSyCLyFFcIcnkLO8e2bfGF_RvniQ5mkbqlaI")

# response = client.models.generate_content(
#     model="gemini-2.0-flash", contents="dado o texto a seguir retorne uma das tres opcoes fenda, alicate, tesoura: Neuro, me de uma ferrmante para cortar"
# )
# print(response.text)
import copy
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import wave
import time
from google import genai
from google.genai import types
import threading

from gtts import gTTS
import os
import tempfile
import subprocess

def translateRequest(request : str):
    if("vermelha" in request.lower()):
        return "pick the red screwdriver and hold it in the air"
    elif("amarela"  in request.lower()):
        return "pick the yellow screwdriver and hold it in the air"
    elif("azul"  in request.lower()):
        return "pick the blue screwdriver and hold it in the air"

def speak_text_file(text, lang='pt'):
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_file = fp.name

    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", audio_file], check=True)
    except FileNotFoundError:
        print("ffplay not found. Install it with: sudo apt install ffmpeg")
    finally:
        os.remove(audio_file)
def record_audio(filename,  duration):
        samplerate=44100
        print("Recording...")
        frames = []
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()  # Wait until the recording finishes

        # Append the recorded audio frames
        frames.append(audio_data)

        # Save the recorded audio to a .wav file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit depth
            wf.setframerate(samplerate)
            wf.writeframes(np.concatenate(frames).tobytes())

        print(f"Audio saved as: {filename}")

class VoiceIA:
    def __init__(self):
        self.client = genai.Client(api_key="AIzaSyCLyFFcIcnkLO8e2bfGF_RvniQ5mkbqlaI")
        self._thread = None
        self._running = False
        self._request= None

    def _listen_loop(self):
        while self._running:
            self._request = self.listen_for_activation_word(activation_word="Neuro ativar acordar")

    def read_request(self):
        request = copy.deepcopy(self._request)
        self._request = None
        return request

    def request_is_ready(self):
        return self._request is not None

    def listen(self):
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._listen_loop, daemon=True)
            self._thread.start()
            print("Listener started.")
        else:
            print("Listener already running.")

    def send_to_gemini(self):
        with open('audio_after_activation.wav', 'rb') as f:
            audio_bytes = f.read()

        response = self.client.models.generate_content(
        model='gemini-2.0-flash',
        contents=[
            'Dado o audio a seguir de uma resposta sempre contendo apenas uma palavra dentre as seguintes que mais se encaixa com o pedido: "fenda amarela", "fenda vermelha", "phillips azul" ou "opção não disponível"',
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type='audio/wav',
            )
        ]
        )

        return (response.text)

    # Function to continuously listen for the activation word
    def listen_for_activation_word(self, activation_word="Neuro hello activate noodle"):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        print(f"Listening for activation word '{activation_word}'...")

        with microphone as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")

            # Listen to the audio and convert to text
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio, language='pt-BR')
                print(f"Heard: {command}")

                # Check if the activation word is in the recognized text
                command = command.split()
                for c in command:
                    if c.lower() in activation_word.lower().split():
                        print(f"Activation word '{c}' detected!")
                        speak_text_file("Olá!")
                        record_audio(filename="audio_after_activation.wav", duration=4)  # Record for 3 seconds
                        speak_text_file("Um momento!")
                        answer= self.send_to_gemini()
                        if("não" in answer.lower()):
                            speak_text_file(answer)
                            return None
                        answer=translateRequest(answer)
                        print(answer)
                        return answer
                if c.lower() == "pegar" or c.lower() == "pega" :
                    speak_text_file("Pegando!")
                    return "Pegar"
            except sr.WaitTimeoutError:
                print("Timeout reached without hearing the activation word.")
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError:
                print("Speech recognition service error.")
            return None

    def listen_to_drop(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()
        activation_word="Soltar Largar Dar"
        print(f"Listening for activation word '{activation_word}'...")

        with microphone as source:
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source)
            print("Listening...")

            # Listen to the audio and convert to text
            try:
                audio = recognizer.listen(source, timeout=5)
                command = recognizer.recognize_google(audio, language='pt-BR')
                print(f"Heard: {command}")

                # Check if the activation word is in the recognized text
                command = command.split()
                for c in command:
                    if c.lower() in activation_word.lower().split():
                        speak_text_file("Soltando")
                        time.sleep(0.5)
                        return True
            except sr.WaitTimeoutError:
                print("Timeout reached without hearing the activation word.")
            except sr.UnknownValueError:
                print("Sorry, I did not understand that.")
            except sr.RequestError:
                print("Speech recognition service error.")
            return False

    def stop(self):
        if self._running:
            self._running = False
            self._thread.join()
            print("Listener stopped.")
        else:
            print("Listener not running.")

def prepare_system():
    return VoiceIA()

if __name__=="__main__":
    VoiceIA().listen()
    try:
        while True:
            pass
    except KeyboardInterrupt:
        pass
