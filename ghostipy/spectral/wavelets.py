import numpy as np
from abc import ABC, abstractmethod
from numba import njit
from scipy.signal import correlate

__all__ = ['Wavelet',
           'MorseWavelet',
           'AmorWavelet',
           'BumpWavelet']

class Wavelet(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def freq_domain_numba(self, omega, scales, out, *, derivative=False):
        pass

    @property
    @abstractmethod
    def is_analytic(self):
        pass

def reference_coi(psifn, reference_scale, *, threshold=1/(np.e**2)):

    omega = np.fft.fftfreq(2**22) * 2 * np.pi
    psif = psifn(omega, reference_scale).squeeze()
    psit = np.fft.ifftshift(np.fft.ifft(psif))
    psit_power = psit.real**2 + psit.imag**2

    power_acs = correlate(psit_power, psit_power)
    maxind = np.argmax(power_acs)
    inds = np.argwhere(power_acs < threshold * np.max(power_acs)).squeeze()
    mask = inds > maxind
    inds = inds[mask]

    reference_coi = inds[0] - maxind

    return reference_coi

def coi(scales, reference_scale, reference_coi, *, fs=1):
    
    scales = np.atleast_1d(scales)
    factors = scales / reference_scale
    print("Reference coi: {} samples".format(reference_coi))
    return factors * reference_coi / fs

@njit
def _morse_freq_domain(omega, scales, gamma, beta, out,
                       *, derivative=False):
    
    # out better be initialized to zeros!
    log_a = np.log(2) + (beta/gamma) * (1+np.log(gamma) - np.log(beta))

    H = np.zeros_like(omega)
    H[omega > 0] = 1

    for ii in range(scales.shape[0]):
        x = scales[ii] * omega
        log_psif = log_a + beta * np.log(np.abs(x)) - np.abs(x)**gamma
        out[ii] = np.exp(log_psif) * H
        if derivative:
            out[ii] *= 1j * omega

    return True

class MorseWavelet(Wavelet):
    
    def __init__(self, *, gamma=3, beta=20):
        
        super().__init__()
        self.gamma = gamma
        self.beta = beta
        self.wp = np.exp( (np.log(self.beta) - np.log(self.gamma)) / self.gamma )
        
    def freq_domain(self, omega, scales, *, derivative=False):
        
        gamma = self.gamma
        beta = self.beta

        scales = np.atleast_1d(scales)
        x = scales[:, None] * omega
        
        H = np.zeros_like(omega)
        H[omega > 0] = 1
        
        with np.errstate(divide='ignore'):
            log_a = np.log(2) + (beta/gamma) * (1+np.log(gamma) - np.log(beta))
            psifs = np.exp(log_a + beta * np.log(np.abs(x)) - np.abs(x)**gamma) * H

        if derivative:
            return 1j * omega * psifs
        
        return psifs
    
    def freq_domain_numba(self, omega, scales, out, *, derivative=False):
        return _morse_freq_domain(omega,
                                  scales,
                                  self.gamma,
                                  self.beta,
                                  out,
                                  derivative=derivative)
    
    def freq_to_scale(self, freqs):
        
        # input should be in normalized radian frequencies
        return self.wp / np.atleast_1d(freqs)
    
    def scale_to_freq(self, scales):
        return self.wp / np.atleast_1d(scales)
    
    def reference_coi(self, *, threshold=1/(np.e**2)):
        base_scale = self.wp / 1
        base_coi = reference_coi(self.freq_domain, base_scale)
        
        return base_scale, base_coi
        
    def coi(self, scales, reference_scale, reference_coi, *, fs=1):
        
        return coi(scales, reference_scale, reference_coi, fs=fs)
    
    @property
    def admissibility_constant(self):
        
        def c_psi(omega):
            
            return self.freq_domain(omega, 1)[0] / omega
        
        return quad(c_psi, 0, np.inf)[0]
    
    @property
    def is_analytic(self):
        return True
        
        
@njit
def _amor_freq_domain(omega, scales, w0, out, *, derivative=False):
    
    H = np.zeros_like(omega)
    H[omega > 0] = 1
    
    for ii in range(scales.shape[0]):
        x = scales[ii] * omega
        out[ii] = 2 * np.exp(-(x - w0)**2/2) * H
        if derivative:
            out[ii] *= 1j * omega
            
    return True
        
        
class AmorWavelet(Wavelet):
    
    def __init__(self, *, w0=7):
    
        self.w0 = w0
        
        
    def freq_domain(self, omega, scales, *, derivative=False):
        w0 = self.w0
        
        scales = np.atleast_1d(scales)
        x = scales[:, None] * omega
        
        H = np.zeros_like(omega)
        H[omega > 0] = 1
        
        psifs = 2 * np.exp(-(x - w0)**2 / 2) * H
        
        if derivative:
            return 1j * omega * psifs
        
        return psifs
    
    def freq_domain_numba(self, omega, scales, out, *, derivative=False):
        return _amor_freq_domain(omega,
                                 scales,
                                 self.w0,
                                 out,
                                 derivative=derivative)
    
    def freq_to_scale(self, freqs):
        
        return self.w0 / freqs
    
    def scale_to_freq(self, scales):
        return self.w0 / np.atleast_1d(scales)
    
    def reference_coi(self, *, threshold=1/(np.e**2)):
        base_scale = self.w0 / 1
        base_coi = reference_coi(self.freq_domain, base_scale)
        
        return base_scale, base_coi
        
    def coi(self, scales, reference_scale, reference_coi, *, fs=1):
        
        return coi(scales, reference_scale, reference_coi, fs=fs)
    
    @property
    def admissibility_constant(self):
        
        def c_psi(omega):
            
            return self.freq_domain(omega, 1)[0] / omega
        
        # sometimes have trouble integrating, so we extend bounds
        # to include negative frequencies. This adds a negligible
        # amount to the final result
        return quad(c_psi, -np.inf, np.inf)[0]
    
    @property
    def is_analytic(self):
        return True
    
@njit
def _bump_freq_domain(omega, scales, mu, sigma, out, *, derivative=False):
    
    for ii in range(scales.shape[0]):
        w = (scales[ii] * omega - mu) / sigma
        tmp = 2 * np.exp( 1 - 1/(1 - w*w) )
        tmp[~(np.abs(w) < 1)] = 0
        tmp[~np.isfinite(tmp)] = 0
        out[ii] = tmp
        if derivative:
            out[ii] *= 1j * omega

    return True
    
    
class BumpWavelet(Wavelet):
    
    def __init__(self, *, mu=5, sigma=0.6):
        
        self.mu = mu
        self.sigma = sigma
        
    def freq_domain(self, omega, scales, *, derivative=False):

        scales = np.atleast_1d(scales)
        
        mu = self.mu
        sigma = self.sigma
        
        w = (scales[:, None] * omega - mu) / sigma
        mask = np.abs(w) < 1
        
        psifs = np.zeros_like(w)
        psifs[mask] = 2 * np.exp( 1 - 1/(1 - w[mask]*w[mask]) )
        
        if derivative:
            return 1j * omega * psifs
        
        return psifs
    
    def freq_domain_numba(self, omega, scales, out, *, derivative=False):
        return _bump_freq_domain(omega, scales, self.mu, self.sigma, out,
                                 derivative=derivative)
    
    def freq_to_scale(self, freqs):
        
        return self.mu / freqs
    
    def scale_to_freq(self, scales):
        return self.mu / np.atleast_1d(scales)
    
    def reference_coi(self, *, threshold=1/(np.e**2)):
        base_scale = self.mu / 1
        base_coi = reference_coi(self.freq_domain, base_scale)
        
        return base_scale, base_coi
        
    def coi(self, scales, reference_scale, reference_coi, *, fs=1):
        
        return coi(scales, reference_scale, reference_coi, fs=fs)
    
    @property
    def admissibility_constant(self):
        
        def c_psi(omega):
            
            return self.freq_domain(omega, 1)[0] / omega
        
        return quad(c_psi, self.mu - self.sigma, self.mu + self.sigma)[0]
    
    @property
    def is_analytic(self):
        return True