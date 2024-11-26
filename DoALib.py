import numpy as np
from scipy.signal import find_peaks
from typing import Union

def linearToDecibels(linear: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 10 * np.log10(linear)

def decibelsToLinear(decibels: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 10 ** (decibels / 10)

#Function to calculate the Root Mean Squared Error
def getRSME(y_true:np.array, y_pred:np.array, realizations):
    return 1/np.sqrt(realizations) * np.sqrt(np.mean((y_true - y_pred) ** 2))

def getNoiseSubspace(x_t,n_multipaths):
    #Calculating the correlation matrix
    correlation_matrix = np.dot(x_t, x_t.conj().T) / n_multipaths

    #Calculating the eigenvalues and eigenvectors
    r_eigvals, r_eigvec = np.linalg.eig(correlation_matrix)

    #Sorting the eigenvalues and eigenvectors
    r_eigval_sorted = r_eigvals.argsort()[::-1]
    r_eigvec_sorted = r_eigvec[:, r_eigval_sorted]

    #Calculating the noise subspace
    noise_subspace = r_eigvec_sorted[:, n_multipaths:]
    
    return noise_subspace

def getDegreesPeaks(power_spectrum, degrees, n_multipaths):
    #Getting two peaks of the power spectrum and your indexes sorted
    peak_indexes, _ = find_peaks(power_spectrum)
    degrees_peaks = []
    power_spectrum_peaks = power_spectrum[peak_indexes]
    #find the two biggest peaks degrees
    for _ in range(n_multipaths):
        index = np.argmax(power_spectrum_peaks)
        power_spectrum_peaks[index] = 0
        degrees_peaks.append(degrees[peak_indexes[index]])
    return degrees_peaks

def getPowerSpectrumMusic(n_antennas, noise_subspace):
    degrees = np.linspace(-np.pi/2, np.pi/2, 1000)
    p_spectrum = np.zeros(degrees.shape)
    for i, degree in enumerate(degrees):
        steering_v = np.exp(1j * np.arange(n_antennas) * (-np.pi * np.sin(degree)))

        num = np.abs(np.dot(np.conj(steering_v.T), steering_v))**2
        den = np.abs(np.dot(np.conj(steering_v.T), np.dot(noise_subspace, np.dot(np.conj(noise_subspace.T), steering_v))))**2
        
        p_spectrum[i] = num / den if den != 0 else 0
    return p_spectrum, degrees

import numpy as np

def getPeaksOfPowerSpectrum(p_spectrum, degrees):
    # Compute the first derivative
    derivative = np.gradient(p_spectrum)
    # Find where the derivative changes sign (positive to negative)
    sign_change = (np.diff(np.sign(derivative)) < 0)
    
    # Add one to indices to account for np.diff reducing length by 1
    potential_peaks = np.where(sign_change)[0] + 1
    
    # Validate that these are actual peaks (greater than neighbors)
    peaks = [i for i in potential_peaks if p_spectrum[i] > p_spectrum[i-1] and p_spectrum[i] > p_spectrum[i+1]]
    
    # Get the degrees corresponding to the peaks
    peaks_degrees = [degrees[i] for i in peaks]
    
    return peaks_degrees

