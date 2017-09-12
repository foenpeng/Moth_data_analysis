from scipy.signal import butter, lfilter

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

if __name__ == "__main__":
    import csv
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import freqz
    from scipy import stats
    from scipy import integrate
    lowcut = 5
    highcut = 75
    fs = 1000
        
    plt.figure(1)
    plt.clf()
    for order in [4, 5, 6]:
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        w, h = freqz(b, a, worN=2000)
        plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

    plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
             '--', label='sqrt(0.5)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.grid(True)
    plt.legend(loc='best')
    
    # Filter a noisy signal.

    with open("n_data.csv", "r") as z_file:
      z_read = csv.reader(z_file)
      line = [[float(item[0]),float(item[1])] for item in z_read]
    line = np.array(line)
    t = line[:,1]
    x = line[:,0]
    mode = stats.mode(x)
    x = x - mode[0]
    plt.figure(2)
    plt.clf()
    plt.plot(t, x, label='Noisy signal')

    y = butter_bandpass_filter(x, lowcut, highcut, fs, order=3)
    #plt.plot(t, y, label='Filtered signal')
    plt.xlabel('time (seconds)')
    plt.grid(True)
    plt.axis('tight')
    plt.legend(loc='upper left')
    
    yint = integrate.cumtrapz(y, t, initial=0)
    displacement = integrate.cumtrapz(yint, t, initial=0)
    plt.figure(3)
    plt.plot(t,displacement)
    
    plt.figure(4)
    plt.clf()
    sp = np.fft.fft(x)
    freq = np.fft.fftfreq(t.shape[-1])
    
    #sp = np.fft.fft(y)
    plt.plot(freq, sp.real, freq, sp.imag)

    plt.show()
