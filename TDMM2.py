import numpy as np
import scipy.stats as st

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.set_bandwidth.html
# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python
def CreateKDEFrom2DData(Xs, Ys, mult = 1.):
    values = np.vstack([Xs, Ys])
    #TODO implement
    kdeo = st.gaussian_kde(values)
    kdeo.set_bandwidth(bw_method=kdeo.factor * mult)
    return kdeo

def Norm(arr):
    return arr/np.sum(arr)

def SampleAxis(xval, ysamples, f):
    vals = np.vstack([np.repeat(np.array([xval])[:,np.newaxis], len(ysamples), 1), ysamples])
    return np.reshape(f(vals).T, ysamples.shape)

class TDMM:
    def __init__(self, Quant=.1, Bandwidth = 1.):
        self.Quant = Quant
        self.BandWidth = Bandwidth
    
    def FromData(self, timeseries):
        Xs = []
        Ys = []

        for i in range(len(timeseries) - 1):
            Xs.append(timeseries[i])
            Ys.append(timeseries[i + 1])

        Xs, Ys = np.array(Xs), np.array(Ys)

        KDEgen = CreateKDEFrom2DData(Xs, Ys, self.BandWidth)

        dataMin = min(timeseries)
        dataMax = max(timeseries)

        self.kdo = KDEgen
        self.Sampled = np.arange(dataMin, dataMax, self.Quant)

    def Step(self, prevVal):
        return np.random.choice(self.Sampled, 1, p=Norm(SampleAxis(prevVal, self.Sampled, self.kdo)))[0]
