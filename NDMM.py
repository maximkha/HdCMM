import numpy as np
import scipy.stats as st

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.set_bandwidth.html
# https://stackoverflow.com/questions/30145957/plotting-2d-kernel-density-estimation-with-python

def Norm(arr):
    return arr/np.sum(arr)

def SampleAxis(otherAxisValues, ysamples, f):
    dims = []
    
    for axisVal in otherAxisValues:
        dims.append(np.repeat(np.array([axisVal])[:,np.newaxis], len(ysamples), 1))
    
    dims.append(ysamples)

    #vals = np.vstack([np.repeat(np.array([xval])[:,np.newaxis], len(ysamples), 1), ysamples])
    dist = np.reshape(f(np.vstack(dims)).T, ysamples.shape)
    return dist

class NDMM:
    def __init__(self, Quant=.1, Bandwidth = 1., N=1):
        self.Quant = Quant
        self.BandWidth = Bandwidth
        self.ND = N
    
    def FromData(self, timeseries):
        pointData = np.zeros((self.ND + 1, len(timeseries)))

        for i in range(len(timeseries) - self.ND):
            for j in range(self.ND + 1):
                pointData[j, i] = timeseries[i + j]

        kdeo = st.gaussian_kde(pointData)
        kdeo.set_bandwidth(bw_method=kdeo.factor * self.BandWidth)
        #kdeo.set_bandwidth(bw_method='silverman') #kdeo.set_bandwidth(bw_method=)

        dataMin = min(timeseries)
        dataMax = max(timeseries)

        self.kdo = kdeo
        self.Sampled = np.arange(dataMin, dataMax, self.Quant)

    def Distribution(self, prevVal):
        if type(prevVal) == float or type(prevVal) == np.float64:
            prevVal = [prevVal]
        if len(prevVal) != self.ND:
            raise ValueError("Prev val should have " + str(self.ND) + " dimensions!")
        return self.Sampled, SampleAxis(prevVal, self.Sampled, self.kdo)

    def Step(self, prevVal):
        x, y = self.Distribution(prevVal)
        return np.random.choice(x, 1, p=Norm(y))[0]
    
    def MaxProb(self, prevVal):
        x, y = self.Distribution(prevVal)
        return x[np.argmax(y)]