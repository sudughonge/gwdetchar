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
	if fit_imcf:
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
	if fit_imcf:
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
		if detrend == 'constant'
			z = np.polyfit(x,y,0)
		else:
			z = np.polyfit(x,y,1)
		return np.poly1d(z)
			

def write_histogram(fname, bin_centers=None, trigger_hist=None, vco_hist=None):
    with open('%s.dat'%fname, 'w') as f:
        for b,t,v in zip(bin_centers, trigger_hist, vco_hist):
            f.write('%4.8f,%4.8f,%4.8f\n' % (b,t,v))


def plot_trigger_histogram(fname ,bin_centers, trigger_hist, time=1, label=None):
    plt.figure(figsize=(12, 6))
		plt.errorbar(bin_centers*1e-3, trigger_hist/time,  yerr=np.sqrt(trigger_hist) / time, drawstyle='steps-mid', label=label)
		
    plt.xlabel('VCO Value [kHz from DC trend]', fontsize=20)
    if time == 1:
        plt.ylabel('Counts')
    else:
        plt.ylabel('Rate [Hz]', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.savefig('%s.png'%fname)
    
    
def iterative_fitting(trigger_hist, vco_hist, bin_centers, TOTTIME, chan='TEST',
                st=None, et=None, directory=None):
    resids = []
    resids_errs = []
    stopping_criteria = [1e6]
    fit_vco_hist = vco_hist[trigger_hist > 0].copy()
    fit_trig_hist = trigger_hist[trigger_hist > 0].copy()
    iters = 0
    resid_diff = 1e6
    binidxs = np.arange(vco_hist[trigger_hist > 0].size)
    newbincs = bin_centers[trigger_hist > 0].copy()
    newvco_hist = vco_hist[trigger_hist > 0].copy()
    newtrig_hist = trigger_hist[trigger_hist > 0].copy()

    bincs_removed = []
    rates = []
    rate_errs = []

    # while resid_diff > .001:
    while stopping_criteria[-1] > 1:
        iters += 1
        weights = 1/np.sqrt(fit_trig_hist)
        rs = []
        # do the fit 100 times, dithering bin heights by gaussian value N(0, sqrt(Nbins))
        # this gives us a better estimate of the residual...otherwise we never really hit the 
        # stopping criterion
        for ii in range(100):
            out = np.polyfit(fit_vco_hist, fit_trig_hist + np.sqrt(fit_trig_hist) * np.random.randn(fit_trig_hist.size), 1, w=weights, full=True)
            rs.append(out[1])
        resids_errs.append(np.std(rs) / 10 / fit_vco_hist.size)
        resids.append(np.mean(rs) / fit_vco_hist.size)
        try:
            stopping_criteria.append(np.sum((np.array(resids[-2]) - np.array(resids[-1]))
                                 / (np.sqrt(np.array(resids_errs[-2])**2 + np.array(resids_errs[-1])**2))))
        except:
            pass
        p,cov = np.polyfit(fit_vco_hist, fit_trig_hist, 1, w=weights, cov=True)
#         out = np.polyfit(fit_vco_hist, fit_trig_hist, 1, w=weights, full=True)
        p = out[0]
        try:
            resid_diff = resids[-2] - resids[-1]
        except:
            resid_diff = 1e6
        rates.append(p[0] * np.sum(vco_hist) / TOTTIME)
        rate_errs.append(np.sqrt(cov[0,0]) * np.sum(vco_hist) / float(TOTTIME))
        fit_hist = fit_vco_hist * p[0] + p[1]
        diff = fit_trig_hist - fit_hist
        bins = ~(diff == max(diff))
        fit_vco_hist = fit_vco_hist[bins]
        fit_trig_hist = fit_trig_hist[bins]
        bincs_removed.append(binidxs[~bins][0])
        binidxs = binidxs[bins]

    weights = 1/np.sqrt(fit_vco_hist)
    p, cov = np.polyfit(fit_vco_hist, fit_trig_hist, 1, w=weights, cov=True)


    bg_rate = p[0] * np.sum(newvco_hist) / TOTTIME
    bg_rate_err = np.sqrt(cov[0,0]) * np.sum(newvco_hist) / float(TOTTIME)

    # get contiguous whistles
    bws = bin_centers[1:] - bin_centers[:-1]
    bw = bin_centers[2] - bin_centers[1]
    segs = SegmentList()
    for ii in range(0, np.size(bincs_removed)):
        seg = Segment([np.mean([newbincs[bincs_removed[ii]], newbincs[bincs_removed[ii]-1]]),
                           np.mean([newbincs[bincs_removed[ii]], newbincs[bincs_removed[ii]+1]])])
        segs.append(seg)
    segs.coalesce()
    wrates = []
    whistle_centers =[]
    wrates_err = []
    whistle_width = []
    nbins = []
    for seg in segs:
        mask = (newbincs > (seg[0] - bw/4.)) * (newbincs < (seg[1] + bw/4.))
        print(mask)
        wrates.append(np.sum(newtrig_hist[mask]) / TOTTIME)
        wrates_err.append(np.sqrt(np.sum(newtrig_hist[mask])) / TOTTIME)
        print(np.sum(newtrig_hist[mask]))
        whistle_centers.append(np.mean(seg))
        whistle_width.append(seg[1] - seg[0])
        nbins.append(0)

    # get rid
    whistle_centers.append('BACKGROUND')
    whistle_width.append('BG')
    nbins.append(0)
    wrates.append(bg_rate)
    wrates_err.append(bg_rate_err)


    print(wrates)
    print(wrates_err)
    data = [whistle_centers, wrates, wrates_err,nbins,
            whistle_width]
    names = ['central frequency', 'rates [Hz]', 'rate errors [Hz]',
             'nbins', 'bin width [kHz]']

    results_tab = Table(data, names=names)
    results_tab.write(generate_generic_name(chan, st, et,
                                            directory, 'RESULTS-TABLE',
                                            'txt'), overwrite=True,
                      format='ascii')
    whistle_hist_fname = generate_generic_name(chan, st, et, directory,
                                               'WHISTLE-RATE', 'png')
    whistle_resids_fname = generate_generic_name(chan, st, et, directory,
                                               'WHISTLE-RESIDS-STOPPING', 'png')
    whistle_hist_with_bkrnd = generate_generic_name(chan, st, et, directory,
                                                    'WHISTLE-RATE-FULL-PLOT',
                                                    'png')
    plt.figure(figsize=(12,6))
    plt.step(newbincs, newtrig_hist/TOTTIME, where='mid')
    plt.step(newbincs, (newvco_hist*p[0] + p[1])/TOTTIME, where='mid')
    plt.scatter(newbincs[bincs_removed], newtrig_hist[bincs_removed]/TOTTIME, marker='*', s=256, c='C2')
    for ii, seg in enumerate(segs):
        plt.scatter( seg[0],[0], marker='>', color='C%d' % ii, s=256)
        plt.scatter( seg[1],[0], marker='<', color='C%d' % ii, s=256)
    # plt.xlim(550, 610)
    # plt.xlim(segs[0][0]-bw, segs[-1][-1]+bw)
    plt.savefig(whistle_hist_fname)
    plt.close()

    plt.errorbar(np.arange(np.size(resids)), resids, yerr=np.array(resids_errs), fmt='o', label='Residual')
    plt.yscale('log')
    plt.plot(stopping_criteria, label='Stopping criterion')
    derivative = np.array(resids[1:]) - np.array(resids[:-1])
    # plt.yscale('log')
    # plt.ylim(-0.1, 0.1)
    plt.title('Residual and stopping criterion')
    plt.xlabel('Iteration')
    plt.ylabel('Residual')
    plt.legend()
    plt.savefig(whistle_resids_fname)
    plt.close()

