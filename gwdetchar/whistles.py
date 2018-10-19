# coding=utf-8
# Copyright (C) Duncan Macleod (2015)
#
# This file is part of the GW DetChar python package.
#
# GW DetChar is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GW DetChar is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GW DetChar.  If not, see <http://www.gnu.org/licenses/>.
'''
Utilities for whistles
'''

from gwpy.timeseries import TimeSeries
import numpy as np

def get_vco_timeseries(ifo, gpsstart, gpsend,fit_imcf=True, **kwargs):
	vco_name = '%s:SYS-TIMING_C_FO_A_PORT_11_SLAVE_CFC_FREQUENCY_5'%ifo
	imcf_name = '%s:IMC-F_OUT_DQ'%ifo	
	vco_ts = TimeSeries.get(vco_name, gpsstart , gpsend, **kwargs)
	if(fit_imcf):
		imcf = TimeSeries.get(imcf_name,gpsstart, gpsend, **kwargs)
		roll_values = np.arange(-1024, 1024, 1024/128.)
		resids = []
		slopes = []
		intercepts = []
		imcf_rsamp = imc[::64]
		for roll_value in roll_values:
			# only sample once per second
			new_vco = (vco[::16])[10:-10]
			# fit
			new_imcf = (np.roll(imcf_rsamp.value, int(roll_value))[::256])[10:-10]
			p = np.polyfit(new_imcf, new_vco, 1, full=True)
			resids.append(p[1])
			slopes.append(p[0][0])
			intercepts.append(p[0][1])
		best_lag = np.argmin(resids)	
		best_fit_time = rollback_times[best_lag]
		best_slope = slopes[best_lag]
		best_intercept = intercepts[best_lag]
		vco_pred = imcf_rsamp*best_slope +  best_intercept
		vco_pred.name = '%s:IMC-VCO_PREDICTION'
				
	else:
		# Downsample as the real sampling rate is just 1 Hz
		vco_pred = vco_ts[8::16]
		vco_pred.epoch = gpsstart + 0.5
	
	return vco_ts, vco_pred

def get_trigger_vco_value(vco_pred, peak_times, fit_imcf=True):
	vco_col = np.zeros(len(peak_times))
	if(fit_imcf):
		for i, pt in enumerate(peak_times):
			try:
				vco = vco_pred.value_at(np.round(pt, 2))
			except IndexError as e:
				print 'Failed to fetch value at %f. Trying integer t values..'%np.round(pt, 2)
			try:
				vco = vco_pred.value_at(np.int(pt))
			except IndexError as e:
				print 'Failed to fetch value at %d. Setting to 0'%np.int(pt)
				vco = 0
			vco_col[i] = vco
			
	else:
		for i,pt in enumerate(peak_times):
			st = np.floor(pt - 0.5) + 0.5
			et = np.ceil(pt + 0.5) - 0.5
			try:
				vco_st = vco_pred.value_at(st)
				vco_et = vco_pred.value_at(et)
				xp = np.array([st, et])
				yp = np.array([vco_st, vco_et])
				vco = np.interp(np.round(pt, 2), xp, yp)
			except IndexError as e:
				print 'Failed to get surrounding VCO values to %f. Setting to 0'%(pt)
				vco = 0
			vco_col[i] = vco
			
def vco_detrend(vco_pred, detrend='constant'):
		x = vco_pred.times.value
		x = x
		y = b.value
		if(detrend=='constant')
			z = np.polyfit(x,y,0)
		else:
			z = np.polyfit(x,y,1)
		return np.poly1d(z)
			

def write_histogram(fname, bin_centers=None, trigger_hist=None, vco_hist=None):
    with open('%s.dat'%fname, 'w') as f:
        for b,t,v in zip(bin_centers, trigger_hist, vco_hist):
            f.write('%4.8f,%4.8f,%4.8f\n' % (b,t,v))


def plot_trigger_histogram(binlist, bin_centers, fname, time=1, labels=None, detrend_func):
    plt.figure(figsize=(12, 6))
    
    for ii, bins in enumerate(binlist):
        if labels is None:
            plt.errorbar(bin_centers / 1e3 - 79e3, bins / time,
                         yerr=np.sqrt(bins) / time, drawstyle='steps-mid')
        else:
            plt.errorbar(bin_centers / 1e3 - 79e3, bins / time,
                         yerr=np.sqrt(bins) / time, drawstyle='steps-mid',
                         label=labels[ii])
    plt.xlabel('VCO Value [kHz from 79 MHz]', fontsize=20)
    if time == 1:
        plt.ylabel('Counts')
    else:
        plt.ylabel('Rate [Hz]', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.savefig(fname)
