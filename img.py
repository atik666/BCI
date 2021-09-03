from pymatreader import read_mat
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

import numpy
from matplotlib import pyplot

import pycwt as wavelet
from pycwt.helpers import find
import pandas as pd
import numpy as np
import cv2

data = read_mat('final.mat')

my_df = pd.DataFrame(data['final'])

a = my_df.iloc[0, :]

def cwt_function(data):
  sig = data
  widths = np.arange(1, 31)
  cwtmatr = signal.cwt(sig, signal.ricker, widths)
  pyplot.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
  #pyplot.close('all')
  pyplot.ioff()
  pyplot.gca().set_axis_off()
  pyplot.margins(0,0)
  pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
  pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())
  pyplot.savefig("filename.jpg", bbox_inches = 'tight', pad_inches = 0)
  #re_size=224
  image = cv2.imread("filename.jpg")
  #image = cv2.resize(image, (re_size,re_size), interpolation=cv2.INTER_CUBIC)
  #pyplot.show()

  return image

im=[]
for i in range(2):
    im.append(cwt_function(a[i]))
    
def cwt_function(data):
  dat = data
  t0 = 0
  dt = 0.01
  N = dat.size
  t = numpy.arange(0, N) * dt + t0
  p = numpy.polyfit(t - t0, dat, 1)
  dat_notrend = dat - numpy.polyval(p, t - t0)
  std = dat_notrend.std()  # Standard deviation
  var = std ** 2  # Variance
  dat_norm = dat_notrend / std  # Normalized dataset
  mother = wavelet.Morlet(6)
  s0 = 2 * dt  # Starting scale, in this case 2 * 0.25 years = 6 months
  dj = 1 / 12  # Twelve sub-octaves per octaves
  J = 7 / dj  # Seven powers of two with dj sub-octaves
  alpha, _, _ = wavelet.ar1(dat)  # Lag-1 autocorrelation for red noise
  wave, scales, freqs, coi, fft, fftfreqs = wavelet.cwt(dat_norm, dt, dj, s0, J, mother)
  iwave = wavelet.icwt(wave, scales, dt, dj, mother) * std
  power = (numpy.abs(wave)) ** 2
  fft_power = numpy.abs(fft) ** 2
  period = 1 / freqs
  power /= scales[:, None]
  signif, fft_theor = wavelet.significance(1.0, dt, scales, 0, 
                                           alpha = alpha, significance_level=0.95, wavelet=mother)
  sig95 = numpy.ones([1, N]) * signif[:, None]
  sig95 = power / sig95
 
  pyplot.close('all')
  pyplot.ioff()
  levels = [0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
  pyplot.contourf(t, numpy.log2(period), numpy.log2(power), numpy.log2(levels), extend='both', cmap='hsv')
  extent = [t.min(), t.max(), 0, max(period)]
  pyplot.contour(t, numpy.log2(period), sig95, [-99, 1], colors='k', linewidths=0, extent=extent)
 
  pyplot.gca().set_axis_off()
  pyplot.margins(0,0)
  pyplot.gca().xaxis.set_major_locator(pyplot.NullLocator())
  pyplot.gca().yaxis.set_major_locator(pyplot.NullLocator())
  pyplot.savefig("filename.jpg", bbox_inches = 'tight', pad_inches = 0) 
  re_size=224
  image = cv2.imread("filename.jpg")
  image = cv2.resize(image, (re_size,re_size), interpolation=cv2.INTER_CUBIC)
  cv2.imwrite("filename.jpg", image)
  image = cv2.imread("filename.jpg")
  #pyplot.show()
  
  return np.array(image)