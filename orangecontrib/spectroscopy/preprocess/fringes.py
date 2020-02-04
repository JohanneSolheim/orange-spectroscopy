import numpy as np
from scipy.signal import hilbert
from sklearn.decomposition import TruncatedSVD

import Orange
from Orange.preprocess.preprocess import Preprocess

try:  # get_unique_names was introduced in Orange 3.20
    from Orange.widgets.utils.annotated_data import get_next_name as get_unique_names
except ImportError:
    from Orange.data.util import get_unique_names

from orangecontrib.spectroscopy.data import getx, spectra_mean
from orangecontrib.spectroscopy.preprocess.utils import SelectColumn, CommonDomainOrderUnknowns, \
    interp1d_with_unknowns_numpy, nan_extend_edges_and_interpolate, transform_to_sorted_features

from orangecontrib.spectroscopy.preprocess.fringe.fringe_emsc import FringeEMSC

def interpolate_to_data(other_xs, other_data, wavenumbers):
    # all input data needs to be interpolated (and NaNs removed)
    interpolated = interp1d_with_unknowns_numpy(other_xs, other_data, wavenumbers)
    # we know that X is not NaN. same handling of reference as of X
    interpolated, _ = nan_extend_edges_and_interpolate(wavenumbers, interpolated)
    return interpolated


class Fringe_EMSCFeature(SelectColumn):
    pass


class Fringe_EMSCModel(SelectColumn):
    pass


class _Fringe_EMSC(CommonDomainOrderUnknowns):

    def __init__(self, reference, wnref, wnLower, wnUpper, nFreq, nIter, domain, scaling=True):
        super().__init__(domain)
        print(reference.X.shape)
        reference=reference.X
        self.fringeCorrection = FringeEMSC(reference, wnref, wnLower=wnLower,
                              wnUpper=wnUpper, nFreq=nFreq, nIter=nIter, scaling=scaling)

    def transformed(self, X, wavenumbers):
        newspectra, parameters, residuals = self.fringeCorrection.transform(X, wavenumbers)
        print(newspectra.shape)
        print(parameters.shape)
        newspectra = np.hstack((newspectra, parameters))
        return newspectra


class MissingReferenceException(Exception):
    pass


class Fringe_EMSC(Preprocess):

    def __init__(self, reference, wnLower, wnUpper, nFreq, nIter, scaling=True, output_model=False):
        # the first non-kwarg can not be a data table (Preprocess limitations)
        # ranges could be a list like this [[800, 1000], [1300, 1500]]
        if reference is None:
            raise MissingReferenceException()
        self.reference = reference
        self.wnref = getx(self.reference)
        self.wnLower = wnLower
        self.wnUpper = wnUpper
        self.nFreq = nFreq
        self.nIter = nIter
        self.scaling = scaling
        self.output_model = output_model

    def __call__(self, data):
        # creates function for transforming data
        common = _Fringe_EMSC(reference=self.reference, wnref=self.wnref, wnLower=self.wnLower, wnUpper=self.wnUpper, nFreq=self.nFreq, nIter=self.nIter, scaling=self.scaling, domain=data.domain)
        # takes care of domain column-wise, by above transformation function
        atts = [a.copy(compute_value=Fringe_EMSCFeature(i, common))
                for i, a in enumerate(data.domain.attributes)]
        model_metas = []
        n_badspec = self.nFreq*2
        # Check if function knows about bad spectra
        used_names = set([var.name for var in data.domain.variables + data.domain.metas])
        if self.output_model:
            i = len(data.domain.attributes)
            for o in range(2):
                n = get_unique_names(used_names, "EMSC parameter " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=Fringe_EMSCModel(i, common)))
                i += 1
            for o in range(n_badspec):
                n = get_unique_names(used_names, "EMSC parameter bad spec " + str(o))
                model_metas.append(
                    Orange.data.ContinuousVariable(name=n,
                                                   compute_value=Fringe_EMSCModel(i, common)))
                i += 1
            n = get_unique_names(used_names, "EMSC scaling parameter")
            model_metas.append(
                Orange.data.ContinuousVariable(name=n,
                                               compute_value=Fringe_EMSCModel(i, common)))
        domain = Orange.data.Domain(atts, data.domain.class_vars,
                                    data.domain.metas + tuple(model_metas))
        print(data)
        return data.from_table(domain, data)
