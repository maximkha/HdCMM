from TDMM import TDMM
import numpy as np
import matplotlib.pyplot as plot
from tqdm import tqdm

time = np.arange(0, 20, .01)
amplitude = np.cos(time)

addNoise = lambda x, s: x + np.random.normal(0, s, len(x))

amplitude = addNoise(amplitude, .1)

plot.plot(time, amplitude)
plot.xlabel('time')
plot.ylabel('data')

mmc = TDMM(.01, 1)
mmc.FromData(amplitude[:len(amplitude)//2])

sY = list([amplitude[0]]) #[0.]
for i in tqdm(range(len(amplitude)-1)):
    sY.append(mmc.Step(sY[-1]))

# print(len(sY))
plot.vlines(x=time[len(time)//2], ymin=min(amplitude), ymax=max(amplitude), color="red")

plot.plot(time, sY)
plot.title('Sine wave')
plot.show()

cMSE = lambda p, t: np.sum((p-t)**2)/len(t)
mse = cMSE(sY[len(amplitude)//2:], amplitude[len(amplitude)//2:])
print("MSE:" + str(mse))

fig = plot.figure()
ax = fig.gca()
ax.set_xlim(min(amplitude), max(amplitude))
ax.set_ylim(min(amplitude), max(amplitude))
# Contourf plot
cset = ax.contourf(mmc.Sampled, mmc.Sampled, mmc.DataMap, cmap='Blues')
## Or kernel density estimate plot instead of the contourf plot
#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])
# Contour plot
#cset = ax.contour(mmc.Sampled, mmc.Sampled, mmc.DataMap, colors='k')
# Label plot
#ax.clabel(cfset, inline=1, fontsize=10)
ax.clabel(cset, inline=1, fontsize=10)
ax.set_xlabel('Previous')
ax.set_ylabel('Next')

plot.show()