import ffmpeg
import os
from scipy.io import wavfile
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
# import noisereduce as nr

def find_freq(audio: np.ndarray) -> np.ndarray:
    # convert to freq domain
    freqDomain = np.fft.fft(audio)
    if freqDomain.shape[0] % 2 == 1:
        freqDomain = freqDomain[:-1]
    freqDomain = np.array_split(freqDomain, 2)[0]
    freqDomain = np.absolute(freqDomain)
    freqDomain = np.square(freqDomain)

    # normalize
    avgPower = np.mean(freqDomain)
    freqDomain = np.multiply(freqDomain, 1/avgPower)

    return freqDomain

path = 'IMG_3713.MOV'
newPath = 'line.wav'
ffmpeg.input(path).output(newPath, preset='ultrafast').run(overwrite_output=1)

def convert_to_audio(data: np.ndarray, sampleRate) -> np.ndarray:
    if data.ndim == 2:
        audio = np.add(data[:, 0], data[:, 1])
    else:
        audio = data
    # audio = nr.reduce_noise(y=audio, sr=sampleRate)
    return audio[np.linspace(0, audio.shape[0], 8000*audio.shape[0]//sampleRate, endpoint=False, dtype=int)]

sampleRate, data = wavfile.read(newPath)
audio = convert_to_audio(data, sampleRate)
freq = find_freq(audio)

plt.plot(audio)
plt.show()

plt.plot(np.square(audio))
plt.show()

plt.plot(freq)
plt.show()