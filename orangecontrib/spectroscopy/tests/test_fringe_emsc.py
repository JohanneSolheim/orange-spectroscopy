import unittest

import numpy as np

import Orange
from Orange.data import FileFormat, dataset_dirs

from orangecontrib.spectroscopy.preprocess.fringes import Fringe_EMSC


class TestFringe_EMSC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def locate_dataset(fn):
            return FileFormat.locate(fn, dataset_dirs)

        path2data1 = locate_dataset('emsc/simFringe.csv')

        v = np.loadtxt(path2data1, delimiter=",")

        cls.wn = v[0,:]
        fringe_spectrum = v[1,:].reshape(-1,1).T
        fringe_spectrum = np.vstack((fringe_spectrum,fringe_spectrum))
        reference = v[2,:].reshape(-1,1).T

        nIter = 1
        nFreq = 1
        wnUpper = 6000*100
        wnLower = 3750*100

        print(fringe_spectrum.shape)
        print(cls.wn.shape)
        print(reference.shape)

        domain_reference = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in cls.wn])
        cls.reference = Orange.data.Table(domain_reference, reference)

        domain_spectra = Orange.data.Domain([Orange.data.ContinuousVariable(str(w))
                                         for w in cls.wn])
        cls.spectra = Orange.data.Table(domain_spectra, fringe_spectrum)

        cls.fringeCorrection = Fringe_EMSC(reference=cls.reference,
                                       wnLower=wnLower,
                                       wnUpper=wnUpper, nFreq=nFreq,
                                       nIter=nIter, scaling=True)

    def test_correction(self):
        print('Hello')
        data = self.fringeCorrection(self.spectra)
        return np.nan
