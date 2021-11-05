import copy
from sys import exit
import csv

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
# import mayavi.mlab as may
from scipy.signal import hilbert
import scipy.ndimage.filters as filters
from scipy.optimize import minimize


class Params:
    def __init__(self, amplitudes, frequencies, damping_coeffs, speeds, no_series, resolution, snr):
        self.amplitudes = amplitudes
        self.frequencies =frequencies
        self.damping_coeffs = damping_coeffs
        self.speeds = speeds
        self.S = no_series
        self.N = resolution
        self.snr = snr # signal-to-noise ratio
    
    def __str__(self):
        return str(vars(self))
        
    
class Radon:

    def __init__(self, Data, dwmin, dwmax, ddw, *args, **kwargs):
        
        '''Data can be a file location string,
        a Params class object containing parameters
        for creating model peaks, a matrix of subsequent
        spectra or a first spectrum for fitting'''
        
        transform_bool = False
        peakpicker_bool = False
            
        self.dwmin = None
        self.dwmax = None
        self.ddw = None
        
        self.abs_Radon = None
        self.complex_Radon = None
        self.frequency = None
        self.speed = None
        
        if (type(Data) == str) or ((args[0] if len(args) > 0 else None) == 'file'):
            
            # Import data from file
            Dataspectrum = np.array(list(csv.reader(open(Data), delimiter="\t")))
            
            # Extract scale in ppm
            Scale = Dataspectrum[1:,0].astype('float64')
            self.b1, self.b2 = float(Scale[0]), float(Scale[-1]) # b1 - smallest, b2 - biggest
            
            # Create data matrix (M[number of series][number of spectrum points])
            Dataspectrum = np.delete(np.delete(np.delete(Dataspectrum,0,0),-1,1),0,1)
            Dataspectrum = Dataspectrum.astype('float64')
            Dataspectrum = Dataspectrum.transpose()
            
            # Matrix of series of spectra
            Spectra = hilbert(Dataspectrum,axis=1)
            self.Spectra = np.conj(Spectra)
            self.n = np.shape(self.Spectra)[1]
            self.s = np.shape(self.Spectra)[0]
            self.FID = np.fft.ifft(self.Spectra,self.n,axis=1) # Convert to time domain signal
            self.t = np.linspace(0,1,self.n)
            
            self.Scale = np.linspace(self.b1,self.b2,self.n) # linearize scale
            transform_bool = True
            peakpicker_bool = True
        
        elif (args[0] if len(args) > 0 else None) == 'Params':
            
            #Create FID of model peaks
            FID = np.zeros((Data.S, Data.N), dtype="complex_")
            t = np.linspace(0, 1, Data.N, endpoint=False)
            for i in range(Data.S):
                for k in range(np.shape(Data.amplitudes)[0]):
                    FID[i] = np.add(FID[i], Data.amplitudes[k]*np.e**(
                        (2*np.pi*1j*(Data.frequencies[k]+i*Data.speeds[k]+1j*Data.damping_coeffs[k])*t)
                        ))
                    if Data.noise: # Add random noise
                        FID[i] = np.add(
                            FID[i],
                            Data.snr*np.max(Data.amplitudes)*np.random.uniform(0, 1, Data.N)
                            )
            
            #fix the first point for Fourier Transform
            for i in range(Data.S):
                FID[i][0] /= 2
            
            self.FID = FID
            self.Spectra = np.fft.fft(FID)
            self.n = np.shape(self.Spectra)[1]
            self.s = np.shape(self.Spectra)[0]
            self.t = t
            
            if 'b1' in kwargs and 'b2' in kwargs:
                self.b1, self.b2 = kwargs['b1'], kwargs['b2']
            else:
                self.b1, self.b2 = 0, 1
            self.Scale = np.linspace(self.b1,self.b2,self.n)
            
            transform_bool = True
            peakpicker_bool = True
        
        elif (args[0] if len(args) > 0 else None) == 'Spectra':
            
            self.Spectra = Data
            self.n = np.shape(self.Spectra)[1]
            self.s = np.shape(self.Spectra)[0]
            self.FID = np.fft.ifft(self.Spectra,self.n,axis=1) # Convert to time domain signal
            self.t = np.linspace(0,1,self.n,endpoint=False)
            
            transform_bool = True
            peakpicker_bool = True
            
        elif (args[0] if len(args) > 0 else None) == 'Fitted':
            
            #args[1] needs to be a Radon object from which it inherits
            Params = Data
            R = args[1]
            
            self.n, self.s = R.n, R.s
            self.DW = R.DW
            self.Scale = R.Scale
            self.b1, self.b2 = R.b1, R.b2
            
            self.Spectra, self.complex_Radon = Radon_in_ppm(Params,R)
            self.abs_Radon = np.abs(self.complex_Radon)
    
        if 'inppb' in kwargs:
            if kwargs['inppb'] == True:
                #Convert Radon dimention to spectral points
                self.dwmin = dwmin/1000*self.n/abs(self.b2-self.b1)
                self.dwmax = dwmax/1000*self.n/abs(self.b2-self.b1)
                self.ddw = ddw/1000*self.n/abs(self.b2-self.b1)
        else:
            self.dwmin, self.dwmax, self.ddw = dwmin, dwmax, ddw
            
        if transform_bool == True:
            self.transform()
        if peakpicker_bool == True:
            self.peakpicker()
    
    #Radon Transform
    def transform(self):
        
        #Phase correction
        self.DW = np.arange(self.dwmin,self.dwmax,self.ddw) # domain of rates of change
        p = np.zeros((len(self.DW),self.s,self.n),dtype="complex64")
        a = 2*np.pi*1j*self.t
        for i in range(len(self.DW)):
            b = a*self.DW[i]
            for k in range(self.s):
                p[i][k] = self.FID[k]*np.e**(-b*k)
            
        #"Diagonal" summation
        Pr = np.zeros((len(self.DW),self.n),dtype="complex64")
        for i in range(len(self.DW)):
            for j in range(self.s):
                Pr[i] += p[i][j]
        
        self.freq = np.arange(self.n)
        
        #Fourier Transform
        phat = np.zeros((len(self.DW),self.n),dtype="complex64")
        PR = np.zeros((len(self.DW),self.n))
        phat = np.fft.fft(Pr)
        PR = np.abs(phat)
        
        self.abs_Radon = PR
        self.complex_Radon = phat
       
    
    #Peak Picker
    def peakpicker(self):
        
        neighborhood_size = 10
        data_max = filters.maximum_filter(self.complex_Radon.real,
                                          neighborhood_size)
        maxima = (self.complex_Radon.real == data_max)
        threshold = (self.complex_Radon.real >= .65*np.max(self.complex_Radon.real))
        results = np.where(np.logical_and(maxima,threshold))
        
        dws = self.DW[results[0]]
        ws = self.freq[results[1]]
        
        #Sort from highest peaks to lowest
        ind = np.argsort(self.complex_Radon.real[results])[::-1]
        dws = [dws[i] for i in ind]
        ws = [ws[i] for i in ind]
        # heights = self.complex_Radon.real[results]
        # heights = [heights[i] for i in ind]
        
        self.frequency = ws
        self.speed = dws
        
    
    def trim(self, lower_bound, upper_bound): # range in ppm
    
        chop1, chop2 = int(self.n*abs(self.b1-lower_bound)/
                           abs(self.b1-self.b2)
                           ),int(self.n*abs(self.b1-upper_bound)/
                            abs(self.b1-self.b2))
        self.b1, self.b2 = lower_bound, upper_bound
        self.abs_Radon = self.abs_Radon[:,chop1:chop2]
        self.complex_Radon = self.complex_Radon[:,chop1:chop2]
        self.n = np.shape(self.complex_Radon)[1]
        
        self.Scale = np.linspace(self.b1,self.b2,self.n)
        
        self.Spectra = self.Spectra[:,chop1:chop2]
        self.FID = None
        
        # Refresh array of peaks found
        self.peakpicker()
        
    
    def subtract(self, R):
        
        # print(np.max(self.complex_Radon.real))
        chop1, chop2 = int(self.n*abs(self.b1-R.b1)/
                           abs(self.b1-self.b2)
                           ),int(self.n*abs(self.b1-R.b2)/
                            abs(self.b1-self.b2))
        self.abs_Radon[:,chop1:chop2] -= R.abs_Radon
        self.complex_Radon[:,chop1:chop2] -= R.complex_Radon
        # print(np.max(self.complex_Radon.real))
        
        self.complex_Radon[self.complex_Radon.real < 0] = 0
        
        # Refresh array of peaks found
        self.peakpicker()


    # def Plot_abs(self):
        
    #     # Plotting the absolute value of radon spectrum
    #     fig = plt.figure()
    #     ax = plt.axes()
    #     X, Y = np.meshgrid(self.Scale, self.DW)
    #     ax.pcolor(X,Y,np.abs(self.complex_Radon))

    #     # may.figure()
    #     # X, Y = np.mgrid[0:np.shape(self.abs_Radon)[0],
    #     #                 0:np.shape(self.abs_Radon)[1]]
    #     # may.surf(X, Y, self.abs_Radon,
    #     #          warp_scale=np.min(np.shape(self.abs_Radon))/np.max(self.abs_Radon))

    def plot_real(self,*args,**kwargs):
        
        fig = go.Figure(data=[go.Surface(
            x = self.Scale,
            y = self.DW, #*1000*abs(self.b2-self.b1)/self.n,
            z = self.complex_Radon.real
            )])
        
        fig.update_layout(
            title = kwargs['title'] if 'title' in kwargs.keys()
                                  else None,
            scene = dict(
                xaxis_title='Frequency [ppm]',
                yaxis_title='Speed [ppb/K]',
                zaxis_visible=False),
            )
        
        fig.show(renderer="colab")
        
        # # # Plotting the real value of radon spectrum
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # if 'title' in kwargs:
        #     plt.title('{0}'.format(kwargs['title']))
        # X, Y = np.meshgrid(self.Scale, 1000*self.DW*abs(self.b2-self.b1)/self.n)
        # c = ax.pcolor(X,Y,self.complex_Radon.real)
        # ax.set_xlabel('Chemical shift [ppm]')
        # ax.set_ylabel('Speed [ppb/K]')
        # if 'Peaks' in kwargs:
        #     Peaks = kwargs['Peaks']
        #     for i in range(np.shape(Peaks)[0]):
        #         if 1000*self.dwmin*abs(self.b2-self.b1)/self.n <= Peaks[i][1] <= 1000*self.dwmax*abs(self.b2-self.b1)/self.n:
        #             ax.plot(Peaks[i][0],Peaks[i][1],'r+')
        #             # ax.text(Peaks[i][0],Peaks[i][1],'{0} ppm\n{1} ppb/K'.format(round(Peaks[i][0],3),round(Peaks[i][1],3)))
        # # if np.shape(Peaks)[0] < 7:
        # #     ax.legend(title='Peaks',fontsize='x-small')#,loc='upper left',bbox_to_anchor=(1, 0))
        # plt.colorbar(c) #,ax=[ax],location='left', pad=0.15)
        # # plt.show(block=False)

        # if 'may' in args:
            
        #     may.figure()
        #     X, Y = np.mgrid[0:np.shape(self.complex_Radon.real)[0],
        #                     0:np.shape(self.complex_Radon.real)[1]]
        #     surf_plot = may.surf(X, Y, self.complex_Radon.real,
        #               warp_scale=np.min(np.shape(self.complex_Radon.real))/np.max(self.complex_Radon.real))
        #     surf_plot.actor.actor.scale = (np.shape(self.complex_Radon.real)[1]/np.shape(self.complex_Radon.real)[0], 1.0, 1.0)
        #     # may.axes(surf_plot, color=(.7, .7, .7), ranges=(np.amin(self.DW),np.amax(self.DW),np.amin(self.Scale),np.amax(self.Scale),0,1),
        #     #           ylabel='Chemical shift [ppm]', xlabel='Speed [ppb/K]',
        #     #           x_axis_visibility=True, y_axis_visibility=False, z_axis_visibility=True)
        #     may.show()
        

# Additional functions

def toppm(R,frequency):
    
    'Convert frequency from spectral points to ppm'
    freq_in_ppm = R.Scale[round(frequency)]
    
    return freq_in_ppm


def Plot_first_and_last(R, *args):

    plt.figure()
    ax = plt.subplot(111)
    if (args[0] if len(args) > 0 else None) != -1:
        ax.plot(R.Scale,np.real(R.Spectra[0]),label='1')
    if (args[0] if len(args) > 0 else None) != 1:
        ax.plot(R.Scale,np.real(R.Spectra[-1]),label='-1')
    plt.title("Real part of spectrum")
    ax.legend()
    ax.invert_xaxis()
    plt.xlabel('Chemical shift [ppm]')


def Plot_FIDs(R):
    
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(R.s):
    #     Y = np.ones(R.n)*i
    #     ax.plot(R.Scale,Y,np.real(R.FID[i]), alpha = 1, zorder = -i)
    # ax.set_xlabel('time, s')
    # ax.set_ylabel('series')
    # ax.set_zlabel('arbitrary unit')
    
    # Only first
    fig = plt.figure()
    R.FID[0][0] *=2
    ax1 = fig.add_subplot(211)
    ax1.plot(np.linspace(0,1,R.n),np.real(R.FID[0]))
    print(R.FID[0])
    ax1.title.set_text('Real part of the FID signal')
    ax1.set_yticklabels([])
    ax1.set_xlabel('time, s')
    plt.tick_params(left=False)
    ax2 = fig.add_subplot(212)
    ax2.plot(np.linspace(0,1,R.n),np.imag(R.FID[0]))
    ax2.title.set_text('Imaginary part of the FID signal')
    ax2.set_xlabel('time, s')
    ax2.set_yticklabels([])
    fig.tight_layout(pad=1)
    plt.tick_params(left=False)

def plot_all_series_spectra(R):
    plt.rcParams["figure.figsize"] = (10, 1)
    for i in range(R.s):
        plt.figure()
        plt.plot(R.Scale, np.real(R.Spectra[i]), label='%f'%i)
    
    # fig, axs = plt.subplots(R.s)
    # fig.suptitle('All series spectra')
    # for i in range(R.s):
    #     axs[i].plot(R.Scale, np.real(R.Spectra[i]), label='%f'%i)
    # plt.tight_layout()
    
    # fig = make_subplots(rows=R.s, cols=2)
    # for i in range(0, R.s):
    #     df = pd.DataFrame({'x': R.Scale,
    #                       'y': R.Spectra[i].real})
    #     fig.add_trace(
    #         px.line(df, x='x', y='y'),
    #         row=i, col=1
    #         )
    # fig.show(renderer="colab")
    
    # plt.figure()
    # ax = plt.subplot(111)
    # for i in range(R.s):
    #     ax.plot(R.Scale,np.real(R.Spectra[i]), label='%f'%i)
    # ax.invert_xaxis()

#3D plot of all series
def Plot_spectra_3D(R):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(R.s):
        Y = np.ones(R.n)*i
        ax.plot(R.Scale,Y,
                np.real(R.Spectra)[i],
                alpha = 1, zorder = -i)
    ax.set_xlabel('frequency, 1/s')
    ax.set_ylabel('series')
    ax.set_zlabel('arbitrary unit')
    plt.show()

#Function to fit peaks to spectra
def FitPeaks(R, Peaks, method, *args):
    print('Fitting peaks')
    radius = 256#512#128#
    
    if len(args):
        
        # Function for finding parameters
        def Fit(x0, passed):
            
            Parameters = passed['Parameters']
            R = passed['R']
            
            Afit = Parameters[:,1]
            wfit = Parameters[:,0]
            Bfit = Parameters[:,2]
            dwfit = x0[0:N]
                
            FitParams = Params(Afit,wfit,Bfit,dwfit,R.s,R.n)
            # print(FitParams.A,
            #       FitParams.w,
            #       FitParams.dw,
            #       FitParams.B,
            #       FitParams.N,
            #       FitParams.S)
            cost = R.complex_Radon.real - Radon(FitParams,R.dwmin, R.dwmax, R.ddw,'Fitted', R, b1=R.b1, b2=R.b2).complex_Radon.real
            norm = np.linalg.norm(cost)
            print(norm)
            multiplycost = 1
            # if np.any(cost<0):
            #     multiplycost *= (1+4*(np.sum(np.where(cost[cost<0]))*np.shape(cost[cost<0])[0])**2/norm)**2
            
            return norm*multiplycost
                        
        itr = iter(range(len(args[0]))) #iter(range(1)) #
        for i in itr:
            
            Parameters = args[0].copy()
            
            #Convert ppm to spectral points
            Parameters[:,0] -= R.b1
            Parameters[:,0] *= R.n/abs(R.b2-R.b1)
            
            # # Cut out the area of interest
            lower_bound, upper_bound = max(0,Parameters[i,0]-radius), min(R.n-1,Parameters[i,0]+radius)
            # Check if boudaries are within spectra edges
            if lower_bound == 0: upper_bound = 2*radius - 1
            elif upper_bound == R.n-1: lower_bound = R.n - 2*radius
            
            R_copy = copy.deepcopy(R)
            R_copy.trim(toppm(R,lower_bound),toppm(R,upper_bound))
            Parameters[:,0] -= lower_bound
            Parameters_fit = Parameters[0 <= Parameters[:,0]].copy()
            Parameters_fit = Parameters_fit[R.n > Parameters_fit[:,0]]
            N = np.shape(Parameters_fit)[0] # Number of relevant peaks

            #Convert back to ppm for fitting
            Parameters_fit[:,0] += lower_bound
            Parameters_fit[:,0] *= abs(R.b2-R.b1)/R.n
            Parameters_fit[:,0] += R.b1
            
            passed = {'Parameters':Parameters_fit,'R':R_copy}
            init = round(((R.dwmax-R.dwmin)/2+R.dwmin)*abs(R.b2-R.b1)/R.n,1) # initial fitting value
            x0 = tuple([init]*N) # 1 ppb/K initial
            OptParams = minimize(Fit, x0, args=(passed), method=method, # 'COBYLA' or 'Nelder-Mead'
                                  options={'maxiter': None})#,'xopt': 0.000001})

            # A = OptParams.x[:N]
            dw = OptParams.x[:N]
            
            # Fited object
            Rfit = Radon(Params(Parameters_fit[:,1],Parameters_fit[:,0],Parameters_fit[:,2],dw,R_copy.s,R_copy.n),
                         R_copy.dwmin, R_copy.dwmax, R_copy.ddw,'Fitted', R_copy)
            
            # R_manual = Radon(Params(Parameters_fit[:,1]/np.pi,Parameters_fit[:,0],Parameters_fit[:,2],np.array([-.17/1000/abs(R.b2-R.b1)*R.n]),R_copy.s,R_copy.n),
            #       R_copy.dwmin, R_copy.dwmax, R_copy.ddw, b1=R_copy.b1, b2=R_copy.b2)
            # R_manual.Plot_real(title='Manual')
            # Plot_first_and_last(R_manual)
            
            for j in range(N):
                Peaks = np.append(Peaks, [[Parameters_fit[j][0],1000*dw[j],Parameters_fit[j][2],Parameters_fit[j][1]]], axis=0)
                # Peaks = np.append(Peaks, [[toppm(R,R.frequency[i]),R.speed[i],B[i],A[i]]], axis=0)
            
            if i<4:#i%1 == 0:
                # # Plot fitted Radon spectrum
                # R_copy.Plot_real('may', title='From Data',Peaks=Peaks[-N:])
                Plot_first_and_last(R_copy,1)
                # Rfit.Plot_real('may', title='Fitted',Peaks=Peaks[-N:])
                Plot_first_and_last(Rfit,1)
            
            zero_out = np.intersect1d(np.nonzero(Parameters[:,0] >= 0),np.nonzero(Parameters[:,0] < R_copy.n))
            args[0][zero_out] = 0
            
            # #Subtract peak
            # R.subtract(Rfit)
            # R.Plot_real('may')
            
            if N > 1:
                for j in range(N-1):
                    try:
                        next(itr)
                    except StopIteration:
                        break
    else:
        
        # Function for finding parameters
        def Fit(x0, R):
            
            Afit = x0[0:N]
            wfit = R.frequency
            Bfit = x0[N:2*N]
            dwfit = R.speed
            FitParams = Params(Afit,wfit,Bfit,dwfit,R.s,R.n)
            # print(FitParams.A,
            #       FitParams.w,
            #       FitParams.dw,
            #       FitParams.B,
            #       FitParams.N,
            #       FitParams.S)
            cost = R.complex_Radon.real - Radon(FitParams,R.dwmin, R.dwmax, R.ddw,'Params', b1=R.b1, b2=R.b2).complex_Radon.real
            norm = np.linalg.norm(cost)
            print(norm)
            multiplycost = 1
            # if np.any(cost<0):
            #     multiplycost *= (1+4*(np.sum(np.where(cost[cost<0]))*np.shape(cost[cost<0])[0])**2/norm)**2
            
            return norm*multiplycost

        itr = iter(range(len(R.frequency)))
        for i in itr:
            
            # # Cut out the area of interest
            lower_bound, upper_bound = max(0,R.frequency[0]-radius), min(R.n-1,R.frequency[0]+radius)
            # Check if boudaries are within spectra edges
            if lower_bound == 0: upper_bound = min(2*radius - 1, R.n - 1)
            elif upper_bound == R.n-1: lower_bound = max(R.n - 2*radius, 0)
            
            R_copy = copy.deepcopy(R)
            R_copy.trim(toppm(R_copy,lower_bound),toppm(R_copy,upper_bound))
            N = np.shape(R_copy.frequency)[0] # Number of relevant peaks
            
            x0 = tuple([.1]*N+[3]*N)
            OptParams = minimize(Fit, x0, args=(R_copy), method=method, # 'COBYLA' or 'Nelder-Mead'
                                  options={'maxiter': None})
            
            # print(OptParams.x)
            
            A = OptParams.x[:N]
            B = OptParams.x[N:]
            # # No fitting
            # B = [2,2.5]
            # A = [2,2]
            
            # Fited object
            Rfit = Radon(Params(A,R_copy.frequency,B,R_copy.speed,R_copy.s,R_copy.n),
                          R_copy.dwmin, R_copy.dwmax, R_copy.ddw, 'Params',b1=R_copy.b1, b2=R_copy.b2)
            
            for j in range(N):
                Peaks = np.append(Peaks, [[(R_copy.frequency[j])*abs(R_copy.b2-R_copy.b1)/R_copy.n+R_copy.b1,
                                           R_copy.speed[j]*1000*abs(R_copy.b2-R_copy.b1)/R_copy.n,B[j],A[j]]], axis=0)
                # Peaks = np.append(Peaks, [[toppm(R,R.frequency[i]),R.speed[i],B[i],A[i]]], axis=0)
            
            # # Plot fitted Radon spectrum
            # R_copy.Plot_real('may',title='From Data',Peaks=Peaks[-N:])
            Plot_first_and_last(R_copy,1)
            # Rfit.Plot_real('may',title='Fitted',Peaks=Peaks[-N:])
            Plot_first_and_last(Rfit,1)
    
            R.frequency, R.speed = R.frequency[N:], R.speed[N:] # delete fitted peaks
            
            # #Subtract peak
            # R.subtract(Rfit)
            
            if N > 1:
                for j in range(N-1):
                    try:
                        next(itr)
                    except StopIteration:
                        break
                    
    return Rfit, Peaks, R


def upload_parameters(file):
    
    '''Format:
    Parameters[0] - shift in ppm
    Parameters[1] - Height
    Parameters[2] - Width in Hz
    Parameters[3] - Area'''
    Parameters = np.array(list(csv.reader(open(file), delimiter="\t")))       
    Parameters = np.delete(Parameters,[0,1],0)
    Parameters = np.array([Parameters[i][0].split(',') for i in range(np.shape(Parameters)[0])]).astype('float64')
    sfrq = 700.1012131 # Hz
    Parameters[:,2] /= sfrq*np.pi #Convert to ppm

    return Parameters

#Create a Radon spectrum from fitted parameters
#This process is done in ppm scale, not spectral points
def Radon_in_ppm(Params, R):
    
    def Lorentz(x, A, V, B, dw):
        
        return np.real(-1/(1j)*A*B/(V - (x-dw) + 1j * B))

    freq = R.Scale
    DW = R.DW*abs(R.b2-R.b1)/R.n
    
    Spectra = np.zeros((R.s,R.n),dtype="complex_")
    for i in range(R.s):
        for j in range(np.shape(Params.w)[0]):
            
            Spectra[i] = np.add(Spectra[i],Lorentz(freq,Params.A[j],Params.w[j],Params.B[j],i*Params.dw[j]))
    
    p = np.zeros((len(R.DW),R.s,R.n),dtype="complex64")
    for i in range(len(R.DW)):
        for k in range(R.s):
            for j in range(np.shape(Params.w)[0]):
                
                p[i][k] = np.add(p[i][k],Lorentz(freq,Params.A[j],Params.w[j],Params.B[j],k*(Params.dw[j]-DW[i])))
        
    #Diagonal summation
    complex_Radon = np.zeros((len(R.DW),R.n),dtype="complex64")
    for i in range(len(R.DW)):
        for j in range(R.s):
            complex_Radon[i] += p[i][j]

    return Spectra, complex_Radon
