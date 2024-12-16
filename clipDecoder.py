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

def setup(folder: str) -> None:
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = folder + '/' + file
            newPath = folder + '/' + folder + ' (wav)/' + file[:-4] + '.wav'
            if not file.startswith('.') and not os.path.isfile(newPath):
                ffmpeg.input(path).output(newPath, loglevel='quiet', preset='ultrafast').run(overwrite_output=1)

def convert_to_audio(data: np.ndarray, sampleRate) -> np.ndarray:
    if data.ndim == 2:
        audio = np.add(data[:, 0], data[:, 1])
    else:
        audio = data
    # audio = nr.reduce_noise(y=audio, sr=sampleRate)
    return audio[np.linspace(0, audio.shape[0], 8000*audio.shape[0]//sampleRate, endpoint=False, dtype=int)]

def plot(files: list[np.ndarray]) -> None:
    plots = len(files)
    fig, axs = plt.subplots(plots)
    for i in range(plots):
        axs[i].plot(np.linspace(0, 4000, files[i][1].shape[0]), files[i][1])
    plt.show()

if __name__ == '__main__':
    setup('configurationAudio')
    setup('testAudio')

    # find frequency domain of configuration files
    configFiles = []
    for file in os.listdir('configurationAudio/configurationAudio (wav)'):
        if not file.startswith('.'):
            path = 'configurationAudio/configurationAudio (wav)/' + file
            sampleRate, data = wavfile.read(path)
            audio = convert_to_audio(data, sampleRate)
            configFiles.append((file, find_freq(audio)))

    plot(configFiles)

    for file in configFiles:
        print(file[0])

    # match files
    testFiles = []
    error = 0

    for file in os.listdir('testAudio/testAudio (wav)'):
        if not file.startswith('.'):
            maxMatchingArea = 0
            maxMatchingFile = configFiles[1][0]

            path = 'testAudio/testAudio (wav)/' + file
            sampleRate, data = wavfile.read(path)
            audio = convert_to_audio(data, sampleRate)

            freq = find_freq(audio)
            testFiles.append((file, freq))

            for configFile in configFiles:
                freqFunc = interp1d(np.arange(0, freq.shape[0], 1), freq)
                freqComparator = freqFunc(np.linspace(0, freq.shape[0] - 1, configFile[1].shape[0]))

                matchingArea = np.mean(np.minimum(freqComparator, configFile[1]))
                if matchingArea > maxMatchingArea:
                    maxMatchingArea = matchingArea
                    maxMatchingFile = configFile[0]
            
            print('audio ' + file[:-4] + ' matches with ' + maxMatchingFile[:-4])
            if maxMatchingFile[:-4] != file[:file.find('Test')]:
                error += 1

    errorRate = error/len(testFiles)
    print('accuracy: {0}%'.format(100*(1-errorRate)))
    # plot(testFiles)