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

#samples function on a 2d grid
# starting at minrange at each axis stepping nsteps until it reaches maxrange.
def SampleF(f, minrange, maxrange, quant):
    xy = np.arange(minrange, maxrange+quant, quant)
    xxyy = np.repeat(xy[:,np.newaxis], len(xy), 1)

    #sampled = np.zeros((len(xy), len(xy)))
    positions = np.vstack([xxyy.ravel(), xxyy.T.ravel()])
    sampled = np.reshape(f(positions), xxyy.shape)
    #xx, yy = np.meshgrid(xy, xy) #since we're sampling on the same range on both axis we don't need this!
    # for i in range(len(xy)):
    #     for j in range(len(xy)):
    #         sampled[i, j] = f((xy[i], xy[j]))[0]

    return sampled, xy

def NormRows(mat):
    mato = np.zeros_like(mat)
    for x in range(mat.shape[0]):
        mato[x, :] = mat[x, :] / np.sum(mat[x, :])
    return mato

# https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()
#print(NormRows(np.array([[1.,2.,3.],[3.,2.,1.]])))

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

        sampleMat, xy = SampleF(KDEgen, dataMin, dataMax, self.Quant)
        self.DataMap = NormRows(sampleMat)
        self.Sampled = xy

    def Step(self, prevVal):
        index = find_nearest(self.Sampled, prevVal)
        return np.random.choice(self.Sampled, 1, p=self.DataMap[index,:])
    
    def MaxProb(self, prevVal):
        index = find_nearest(self.Sampled, prevVal)
        return self.Sampled[np.argmax(self.DataMap[index,:])]
