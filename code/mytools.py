import numpy as np
import statistics
import math
import glob
import re
from astropy.io import fits
from scipy import stats
import numpy as np
from scipy.signal import savgol_filter
from scipy import interpolate
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import scipy.signal
from scipy.signal import lombscargle
from scipy.signal import find_peaks
import pandas as pd
from scipy.optimize import curve_fit
import pylab
from dictionary import exo_Dict
import warnings
warnings.filterwarnings('ignore')

class astro_labs():
    
    'This is a class, where all the relevent functions can be performed'
    
    def __init__(self, mykepler):
        
        self.mykepler = mykepler
        
    def get_data(self):
        
        '''
        2.1
        Import data
        Remove NAN values
        Extend data points to one array
        '''
        
        self.F = []
        self.T = []
        self.FE = []
        
        for lcfile in glob.glob('Data/Object%slc/kplr*.fits' %(self.mykepler)):
            tmp = fits.open(lcfile)
            tmptime = (tmp[1].data['TIME'])
            tmpflux = (tmp[1].data['PDCSAP_FLUX'])
            tmperror = (tmp[1].data['PDCSAP_FLUX_ERR'])
            finite_value = np.isfinite(tmpflux)
            f = tmpflux[finite_value]
            t = tmptime[finite_value]
            fe = tmperror[finite_value]
            self.F.extend(f)
            self.T.extend(t)
            self.FE.extend(fe)
                
        return
   
    def filtering(self, Savitzky_Golay):
        
        '''
        2.1
        Filtering using Savitzky Golay
        Filtering using median filter
        Normalistating data 
        Removed outliers using Z score
        '''
        
        if Savitzky_Golay is True:
            interp_savgol = savgol_filter(self.F, window_length=1001, polyorder=3)
            self.f = self.F/interp_savgol
        else:
            testfil = medfilt(self.F, kernel_size = 121)
            self.f = self.F/testfil
            
        z = np.abs(stats.zscore(self.f))
        filtered_entries = (z < 3)

        self.flux_fil = np.array(self.f)[filtered_entries]
        self.time_fil = np.array(self.T)[filtered_entries]
        self.flux_error_fil = np.array(self.FE)[filtered_entries]
        
        sorted_arr = np.argsort(self.time_fil)
        self.flux = self.flux_fil[sorted_arr]
        self.time = self.time_fil[sorted_arr]
        self.flux_error = self.flux_error_fil[sorted_arr]
        
        plt.figure(figsize=(15,5))
        plt.ylabel('Flux [joules/m$^2$/s]')
        plt.xlabel('Time [days]')
        plt.plot(self.time, self.flux, marker='.', ls = 'none', c = 'black')
       
        return self.flux, self.time
    
    def periodogram(self):
        
        '''
        2.2
        Periodogram for frequency
        Periodogram for time
        Determin period of periodogram peaks
        '''
        
        delta_t = self.time[1] - self.time[0]      
        max_step = np.max(self.time) - np.min(self.time)
        freqs = np.linspace((1/max_step), (1/delta_t), 10000)
        
        lomb = scipy.signal.lombscargle(self.time, self.flux, freqs, precenter=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
        fig.suptitle('Lomb-Scargle Periodogram')
        
        ax1.plot(freqs, lomb, c = 'black')
        ax1.set_title('lomb 1')
        ax1.set(xlabel='Frequency (1/days)', ylabel='Lomb-Scargle Power')
        
        period = np.linspace(1,200, 100000)
        lomb2 = scipy.signal.lombscargle(freqs, lomb, period, precenter=True)
        
        ax2.plot(period, lomb2, c = 'black')
        ax2.set_title('lomb 2')
        ax2.set(xlabel='Period (days)', ylabel= 'Lomb-Scargle Power')
        
        left, bottom, width, height = [0.792 , 0.640, 0.1, 0.2]
        ax_new = fig.add_axes([left, bottom, width, height])
        ax_new.plot(period, lomb2, color='black')
        ax_new.set_xlim(0, 50)
        ax_new.set_ylim(0, 1e-12)
        
        peaks = find_peaks(lomb2, height = 0.01e-12)
        height = peaks[1]['peak_heights'] #list containing the height of the peaks
        peak_pos = period[peaks[0]]   #list containing the positions of the peaks
        
        return peak_pos
    
    def quartic_transit(self, t, a, transit_loc, depth, baseline):
        """
        2.3
        A function to describe the shape of a transit, including the in- and egress using a
        quartic
        t: time (or phase)
        a: free parameter that controls the shape of the transit, typically ~>50
        transit_loc: location of the mi-transit point (either in time or phase)
        depth: depth of the transit (e.g. -0.1 for a 10% drop in flux)
        baseline: value of lightcurve outside transit (typically 1)
        returns: f(t) for the transit
        """
        return (a*(t-transit_loc)**4 + 0.5*depth) - abs((a*(t-transit_loc)**4 + 0.5*depth)) + baseline
        
    def transit_analysis(self, period, xlim, transit_loc):
        """
        2.3
        Function to fit to transit
        Perform iteration
        Provide corrected values (popt)
        Provide covarience (pcov)
        """
        # Create a pandats dataframe from the 
        data = pd.DataFrame({'time': self.time, 'flux': self.flux, 'error': self.flux_error})
    
        # Create the phase 
        data['phase'] = data.apply(lambda x: ((x.time/ period) - np.floor(x.time / period)), axis=1)
    
        # Creates the out phase, flux and error
        self.phase_conc = np.concatenate((data['phase'], data['phase'] + 1.0, data['phase'] + 2.0))
        self.flux_conc = np.concatenate((self.flux, self.flux, self.flux))
        self.err_conc = np.concatenate((self.flux_error, self.flux_error, self.flux_error))
        self.xlim = xlim
        mask = (self.xlim[1]>self.phase_conc) & (self.phase_conc>self.xlim[0])
        self.ph_conc = self.phase_conc[mask]
        fl_conc = self.flux_conc[mask]
        err_conc = self.err_conc[mask]
        
        # Creates corrected values and covarience
        popt, pcov = curve_fit(self.quartic_transit, self.ph_conc, fl_conc, sigma=err_conc, p0=[50, transit_loc, -0.0008, 1])
        self.func_conc = self.quartic_transit(self.ph_conc, *popt)
                         
        return popt, pcov
        
    def transit_plots(self):
        """
        2.3
        Produces 2 x 2 subplot of transits
        """
        
        # Introduce data from dictionary for fucntion
        period, xlim, transit_loc = exo_Dict['A']['period'], (exo_Dict['A']['xlim_min'], exo_Dict['A']['xlim_max']), exo_Dict['A']['transit_loc']
        
        # Function to fit transit
        self.A, Apcov = self.transit_analysis(period, xlim, transit_loc)
        
        # Introduce 2 x 2 subplot
        fig, axs = plt.subplots(2,2, figsize=(15,10))
        fig.suptitle('Transit plots of identified exoplanets')
        
        axs[0, 0].scatter(self.ph_conc, self.func_conc, ls='-', label = 'fitted line', lw = 0.01, zorder = 3, c = 'r')
        axs[0, 0].scatter(self.phase_conc, self.flux_conc, marker = '.', ls = 'None', label = 'original data', lw = 1, s = 2, c = 'black')
        axs[0, 0].set_xlim(self.xlim)
        axs[0, 0].set_title('Exoplanet A: ' + "{0:.4g}".format(period))
        
        period, xlim, transit_loc = exo_Dict['B']['period'], (exo_Dict['B']['xlim_min'], exo_Dict['B']['xlim_max']), exo_Dict['B']['transit_loc']
        self.B, Bpcov = self.transit_analysis(period, xlim, transit_loc)
        
        axs[0, 1].scatter(self.ph_conc, self.func_conc, ls='-', label = 'fitted line', lw = 0.01, zorder = 3, c = 'r')
        axs[0, 1].scatter(self.phase_conc, self.flux_conc, marker = '.', ls = 'None', label = 'original data', s = 2, c = 'black')
        axs[0, 1].set_xlim(self.xlim)
        axs[0, 1].set_title('Exoplanet B: ' + "{0:.4g}".format(period))
        
        period, xlim, transit_loc = exo_Dict['C']['period'], (exo_Dict['C']['xlim_min'], exo_Dict['C']['xlim_max']), exo_Dict['C']['transit_loc']
        self.C, Cpcov = self.transit_analysis(period, xlim, transit_loc)
        
        axs[1, 0].scatter(self.ph_conc, self.func_conc, ls='-', label = 'fitted line', lw = 0.01, zorder = 3, c = 'red')
        axs[1, 0].scatter(self.phase_conc, self.flux_conc, marker = '.', ls = 'None', label = 'original data', s = 2, c = 'black')
        axs[1, 0].set_xlim(self.xlim)
        axs[1, 0].set_title('Exoplanet C: ' + "{0:.4g}".format(period))
        
        period, xlim, transit_loc = exo_Dict['D']['period'], (exo_Dict['D']['xlim_min'], exo_Dict['D']['xlim_max']), exo_Dict['D']['transit_loc']
        self.D, Dpcov = self.transit_analysis(period, xlim, transit_loc)
        
        axs[1, 1].scatter(self.ph_conc, self.func_conc, ls='-', label = 'fitted line', lw = 0.01, zorder = 3, c = 'red')
        axs[1, 1].scatter(self.phase_conc, self.flux_conc, marker = '.', ls = 'None', label = 'original data', s = 2, c = 'black')
        axs[1, 1].set_xlim(self.xlim)
        axs[1, 1].set_title('Exoplanet D: ' + "{0:.4g}".format(period))
        
        for ax in axs.flat:
            ax.set(xlabel='Phase', ylabel='Flux')
            
        # To find covarience of delta flux (depth)
        # print('Covarience in depth of A: ' ,np.sqrt(np.diag(Apcov)))
  
        return
    
    def linear_regression(self, phase, period, depth):
        """
        2.5
        Function to identify minimum poins at each period
        Produces regression plot
        Calculates residule
        """
        # Identified first peak
        start_peak = phase*period
        
        # Identified location of transits
        peaks = []
        time_points = []
        
        # Iterate through lightcurve
        for i in range(math.ceil(len(self.time)/period)):
            
            # Determine period peaks
            peaks = start_peak + (period*i)
            
            # Identify a range to locate minima
            upper_lim, lower_lim = (peaks+2, peaks-2)
            
            # Mask transit range
            mask_peak = (upper_lim>self.time) & (lower_lim<self.time)
            masked_flux = self.flux[mask_peak]
            masked_time = self.time[mask_peak]
            
            # Filter out ranges with too few points
            if len(masked_flux) == 0 or len(masked_flux) < 4:
                continue
                
            # Find min flux at each peak by curve fitting
            popt, pcov = curve_fit(self.quartic_transit, masked_time, masked_flux, p0=[10, peaks, depth, 1])
            period_location = popt[1]
            
            # Finding relevant time points
            time_points.append(period_location)
            
        # Subtracting points from each other to gain actual period value at each minima
        difference = np.array([t - s for s, t in zip(time_points, time_points[1:])])
        mask = difference/period < 2
        delta = period - difference[mask]
        index = list(range(len(delta)))
        
        # Error calculation
        # Standard error of residule, see Appendix B
        sum_yi = 0
        sum_xi = 0
        N = len(delta) - 2
        n_hat = sum(delta)/len(delta)
        for i in range(len(delta)):
            sum_yi = delta[i]**2 + sum_yi
            delta_n = index[i] - n_hat
            sum_xi = delta_n**2 + sum_xi
        
        error = np.sqrt(sum_yi/N)/np.sqrt(sum_xi)
        
        # Error
        #print(period, error)
        
        return delta, index
    
    def regression_plots(self):
        """
        2.5
        Produces 2 x 2 subplot of regression
        """
        # Introduce data from dictionary for fucntion
        period = exo_Dict['A']['period']
        phase = self.A[1]
        depth = self.A[2]
        deltaA, indexA = self.linear_regression(phase, period, depth)
        
        fig, axs = plt.subplots(2,2, figsize=(15,10))
        fig.suptitle('Plots of residules')
        
        axs[0, 0].scatter(indexA, deltaA, ls='-', label = 'actual', lw = 0.1, zorder = 3, c = 'black')
        axs[0, 0].axhline(0, c = 'red')
        axs[0, 0].set_title('Exoplanet A: ' + "{0:.4g}".format(period))
        
        period = exo_Dict['B']['period']
        phase = self.B[1]
        depth =  self.B[2]
        deltaB, indexB = self.linear_regression(phase, period, depth)
        
        axs[0, 1].scatter(indexB, deltaB, ls='-', label = 'actual', lw = 0.1, zorder = 3, c = 'black')
        axs[0, 1].axhline(0, c = 'red')
        axs[0, 1].set_title('Exoplanet B: ' + "{0:.4g}".format(period))
        
        period = exo_Dict['C']['period']
        phase = self.C[1]
        depth = self.C[2]
        deltaC, indexC = self.linear_regression(phase, period, depth)
        
        axs[1, 0].scatter(indexC, deltaC, ls='-', label = 'actual', lw = 0.1, zorder = 3, c = 'black')
        axs[1, 0].axhline(0, c = 'red')
        axs[1, 0].set_title('Exoplanet C: ' + "{0:.4g}".format(period))
        
        period = exo_Dict['D']['period']
        phase = self.D[1]
        depth = self.D[2]
        deltaD, indexD = self.linear_regression(phase, period, depth)
        
        axs[1, 1].scatter(indexD, deltaD, ls='-', label = 'actual', lw = 0.1, zorder = 3, c = 'black')
        axs[1, 1].axhline(0, c = 'red')
        axs[1, 1].set_title('Exoplanet D: ' + "{0:.4g}".format(period))
        
        for ax in axs.flat:
            ax.set(xlabel='Index', ylabel='Change in Period')
        return
               