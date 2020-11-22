from TDMM2 import TDMM
from NDMM import NDMM
import numpy as np
import matplotlib.pyplot as plot
from tqdm import tqdm

time = np.arange(0, 20, .01)
amplitude = np.cos(time)

addNoise = lambda x, s: x + np.random.normal(0, s, len(x))

amplitude = addNoise(amplitude, .01)

plot.plot(time, amplitude)
plot.xlabel('time')
plot.ylabel('data')

ndim = 60
mmc = NDMM(.005, .666, ndim)
mmc.FromData(amplitude[:len(amplitude)//2])

sY = list(amplitude[:ndim]) #[0.]
for i in tqdm(range(len(amplitude) - ndim)):
    x = sY[-ndim:]
    if ndim == 0: x = []
    sY.append(mmc.MaxProb(x))

# print(len(sY))
plot.vlines(x=time[ndim], ymin=min(amplitude), ymax=max(amplitude), color="green")
plot.vlines(x=time[len(time)//2], ymin=min(amplitude), ymax=max(amplitude), color="red")

plot.plot(time, sY)
plot.title('Sine wave')
plot.show()

cMSE = lambda p,t: np.sum((p-t)**2)/len(t)
mse = cMSE(sY[len(amplitude)//2:], amplitude[len(amplitude)//2:])
print("MSE:" + str(mse))