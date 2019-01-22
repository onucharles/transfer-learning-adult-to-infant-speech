from scipy import signal
import matplotlib.pyplot as plt
import librosa

def plot_spectrogram():
    plt.figure()
    file_path1 = r'C:\Users\Charley\Dropbox (NRP)\paper\afe0b87d_nohash_0.wav'
    y, sr = librosa.load(file_path1, sr=8000)
    f, t, Sxx = signal.spectrogram(y, sr)
    plt.subplot(121)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("Spectogram of a 'no'")
	
    file_path2 = r'C:\Users\Charley\Dropbox (NRP)\paper\ffd2ba2f_nohash_1.wav'
    y, sr = librosa.load(file_path2, sr=8000)
    f, t, Sxx = signal.spectrogram(y, sr)
    plt.subplot(122)
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title("Spectogram of a 'yes'")
	
    plt.show()


if __name__ == '__main__':
    plot_spectrogram()