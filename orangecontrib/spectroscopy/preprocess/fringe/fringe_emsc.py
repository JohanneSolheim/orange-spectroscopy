import numpy as np
from scipy.fftpack import fft


class FringeEMSC:
    def __init__(self, refSpec, wnref, wnLower, wnUpper, nFreq, nIter, scaling=True):
        self.refSpec = refSpec
        self.wnref = wnref
        self.wnUpper = wnUpper
        self.wnLower = wnLower
        self.nFreq = nFreq
        self.nIter = nIter
        self.scaling = scaling

    def frequency_from_spectrum(self, rawspectrum, wn):
        indLower = np.argmin(abs(wn-self.wnLower))
        indUpper = np.argmin(abs(wn-self.wnUpper))

        region_frequency = rawspectrum[indLower:indUpper]
        # wn_frequency = wn[np.argmin(abs(wn-wnLower)):np.argmin(abs(wn-wnUpper))]

        # Zero pad signal
        nPad = region_frequency.shape[0]
        region_frequency = region_frequency-((np.max(region_frequency)-np.min(region_frequency))/2+np.min(region_frequency))
        region_frequency = np.hstack((np.zeros([nPad]), region_frequency, np.zeros([nPad])))

        # Fourier transform signal
        fTransform = fft(region_frequency)

        # Calculate the frequancy axis in the Fourier domain
        N = region_frequency.shape[0]
        dw = wn[1]-wn[0]
        dx = 2*np.pi/(N*dw)
        x = np.linspace(0.0, N*dx, N+1)

        fTransform = np.abs(fTransform[0:N//2])
        x = x[0:N//2]

        freqMaxInd = np.argpartition(fTransform[3:], -self.nFreq)[-self.nFreq:]  # FIXME implement in more robust way
        freqMax = x[freqMaxInd+3]
        return freqMax  # FIXME give all frequencies for all different spectra

    def setup_emsc(self, freqMax, wn):
        const = np.ones(len(wn))

        N = wn.shape[0]
        c_coeff = 0.5*(wn[0]+wn[N-1])
        m0 = -2.0/(wn[0]-wn[N-1])
        linear = m0*(wn-c_coeff)

        M = np.vstack((const, linear)).T

        for i in range(0, self.nFreq):
            sinspec = np.sin(freqMax[i]*wn)
            sinspec = sinspec.reshape(-1, 1)
            cosspec = np.cos(freqMax[i]*wn)
            cosspec = cosspec.reshape(-1, 1)
            M = np.hstack((M, sinspec, cosspec))
        M = np.hstack((M, self.refSpec.reshape(-1, 1)))
        return M

    def solve_emsc(self, rawspectrum, M):
        n_badspec = self.nFreq*2

        params = np.linalg.lstsq(M, rawspectrum, rcond=-1)[0]
        corrected = rawspectrum

        for x in range(0, 2 + n_badspec):
            corrected = (corrected - (params[x] * M[:, x]))
        if self.scaling:
            corrected = corrected/params[-1]

        residuals = rawspectrum - np.dot(params, M.T)
        return corrected, params, residuals

    def correct_spectra(self, rawspectra, wn):
        newspectra = np.full(rawspectra.shape, np.nan)
        residuals = np.full(rawspectra.shape, np.nan)
        parameters = np.full([rawspectra.shape[0], self.nFreq*2 + 3], np.nan)
        for i in range(0, rawspectra.shape[0]):
            freq = self.frequency_from_spectrum(rawspectra[i, :], wn)
            emsc_mod = self.setup_emsc(freq, wn)
            corr, par, res = self.solve_emsc(rawspectra[i, :], emsc_mod)
            newspectra[i, :] = corr
            residuals[i, :] = res
            parameters[i, :] = par.T
        return newspectra, parameters, residuals

    def transform(self, spectra, wn):
        newspectra = spectra
        for j in range(0, self.nIter):
            newspectra, parameters, residuals = self.correct_spectra(newspectra, wn)
        return newspectra, parameters, residuals

