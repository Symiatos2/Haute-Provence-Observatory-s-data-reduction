#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pydis_um.py: modification of the ap_trace and ap_extract functions from J.R.A
Davenport's MIT licensed PyDIS reduction software for use in the Montpellier and
Lyon Master's projects.

ap_trace modifications: aim to trace a specific aperture on multi-aperture
spectra, e.g. w/ a slicer such as on OHP152/AURELIE :
  * the central spectral bin is used as a starting guess instead of the image
    summed along the wavelength axis
  * the user can define the fit window (ap_min, ap_max) and 1st guess of the
    aperture center (ap_center) w/o interactive window which does not work well
    w/ Jupyter and is less efficient to process a series of spectra
  * ap_center, ap_min, ap_max are defined as lists in order to extract multiple
    traces from a spectrogram, so far only the first element is used
  * starting from the central bin fit the algorithm fits the neighbouring bin
    using the results from the previous one
  * the final trace is not anymore a spline which often results in wiggles but a
    polynomial fit with user-defined order (ofit)
  * print the RMS of the fit residual

ap_extract modifications: can define asymmetric star aperture and sky apertures
around the trace, the following variables are changed:
  * apwidth -> ap_l, ap_u
  * skysep -> skysep_l, skysep_u
  * skywidth -> sky_l, sky_u

_gaus: the helper function is copied here w/o modification

28 Jul 2022: Created w/ the functions _gauss, ap_trace, ap_extract.
"""

__author__ = "Julien Morin <julien.morin@umontpellier.fr>"
__date__ = "2022-07-28"
__version__ = "1.0"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.signal
from scipy.optimize import curve_fit

def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function, for internal use only

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b

def ap_trace(img, fmask=(1,), nsteps=20, ap_center = [], ap_min=[], ap_max=[],
             recenter=False, prevtrace=(0,), bigbox=15, ofit=2,
             Saxis=1, display=False, display_all=False):
    """
    Modification of pydis.ap_trace

    Trace the spectrum aperture in an image

    Assumes wavelength axis is along the X, spatial axis along the Y.
    Chops image up in bins along the wavelength direction, fits a Gaussian
    within each bin to determine the spatial center of the trace. Finally,
    draws a polynomial fit through the bins to up-sample the trace.

    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:

        >>> hdu = fits.open('file.fits')  # doctest: +SKIP
        >>> img = hdu[0].data  # doctest: +SKIP

    nsteps : int, optional
        Keyword, number of bins in X direction to chop image into. Use
        fewer bins if ap_trace is having difficulty, such as with faint
        targets (default is 20, minimum is 5)
    ap_center : list(float), optional
        initial guess for the central aperture fit, allows to select which
        aperture is traced in multi-aperture spectra; default is an empty list
        in which case the main aperture is traced
        :not implemented: defined as a list for multi-aperture extraction but
        only the first element is used so far
    ap_min : list(float), optional
        width of the fitting window below ap_center for the central aperture
        fit, allows to select which aperture is traced in multi-aperture
        spectra; default is an empty list in which case the main aperture is
        traced
        :not implemented: defined as a list for multi-aperture extraction but
        only the first element is used so far
    ap_max : list(float), optional
        width of the fitting window above ap_center for the central aperture
        fit, allows to select which aperture is traced in multi-aperture
        spectra; default is an empty list in which case the main aperture is
        traced
        :not implemented: defined as a list for multi-aperture extraction but
        only the first element is used so far
    fmask : array-like, optional
        A list of illuminated rows in the spatial direction (Y), as
        returned by flatcombine
    recenter : bool, optional
        Set to True to use previous trace, but allow small shift in
        position. Currently only allows linear shift (Default is False)
    bigbox : float, optional
        The number of sigma away from the main aperture to allow to trace
    ofit: int, optional
        Polynomial order of the fit trough the aperture centers (default is 2)
    display : bool, optional
        If set to true display the trace over-plotted on the image
    Saxis : int, optional
        Set which axis the spatial dimension is along. 1 = Y axis, 0 = X.
        (Default is 1)

    Returns
    -------
    my : array
        The spatial (Y) positions of the trace, resulting from a polynomial fit
        over the entire wavelength (X) axis
    """

    # define the wavelength axis
    Waxis = 0
    # add a switch in case the spatial/wavelength axis is swapped
    if Saxis is 0:
        Waxis = 1

    print('Tracing Aperture using nsteps='+str(nsteps))
    nsteps+=1
    # the valid y-range of the chip
    if (len(fmask)>1):
        ydata = np.arange(img.shape[Waxis])[fmask]
    else:
        ydata = np.arange(img.shape[Waxis])

    # need at least 5 samples along the trace. sometimes can get away with very few
    if (nsteps<5):
        nsteps = 5

    # median smooth to crudely remove cosmic rays
    img_sm = scipy.signal.medfilt2d(img, kernel_size=(5,5))

    #--- 1st guess of aperture params
    ztot = img_sm.sum(axis=Saxis)[ydata]
    yi = np.arange(img.shape[Waxis])[ydata]

    # define the X-bin edges
    xbins = np.linspace(0, img.shape[Saxis], nsteps, dtype='int')
    ybins = np.zeros_like(xbins, dtype='float32')
    #--- use the middle bin as a starting condition
    i = int(len(xbins)/2)
    if Saxis is 1:
        zi = np.sum(img_sm[ydata, xbins[i]:xbins[i+1]], axis=Saxis)
    else:
        zi = img_sm[xbins[i]:xbins[i+1], ydata].sum(axis=Saxis)
    #--- Pick the strongest source, good if only 1 obj on slit
    if len(ap_center) == 0:
        peak_y = yi[np.nanargmax(zi)]
        peak_guess = [np.nanmax(zi), np.nanmedian(ztot), peak_y, 2.]
    #--- If apertures 1st guess params are defined use them
    else:
        peak_guess = [zi[ap_center[0]], .5*(zi[ap_min[0]]+zi[ap_max[0]]), ap_center[0], 2.]

    #-- use middle of previous trace as starting guess
    if (recenter is True) and (len(prevtrace)>10):
        peak_guess[2] = np.nanmedian(prevtrace)

    #-- fit a Gaussian to peak
    if len(ap_center)==0:
        popt_0, pcov = curve_fit(_gaus, yi[np.isfinite(zi)], zi[np.isfinite(zi)], p0=peak_guess)
    else:
        popt_0, pcov = curve_fit(_gaus, yi[np.isfinite(zi)][ap_min[0]:ap_max[0]], zi[np.isfinite(zi)][ap_min[0]:ap_max[0]], p0=peak_guess)
    #-- plot central bin, aperture presets, gaussian fit
    if display:
        plt.figure()
        plt.plot(zi)
        plt.plot(_gaus(yi[np.isfinite(zi)], *popt_0))
        if len(ap_center)==0:
            plt.axvline(peak_y , c='tab:orange', ls='--')
        else:
            plt.axvline(ap_center[0], c='tab:orange', ls='--')
            plt.axvline(ap_min[0], c='tab:green', ls=':')
            plt.axvline(ap_max[0], c='tab:green', ls=':')
            plt.axhline(.5*(zi[ap_min[0]]+zi[ap_max[0]]), c='tab:gray', ls=':')
        plt.show()

    popt = popt_0

    #-- from central bin loop toward increasing index
    for i in range(int(len(xbins)/2)+1,len(xbins)-1):
        #print("bin #", i) 
        #-- only allow data within a box around this peak
        #print(popt[2], popt[3], popt[2] - popt[3]*bigbox, popt[2] + popt[3]*bigbox)
        ydata2 = ydata[np.where((ydata>=popt[2] - popt[3]*bigbox) &
                                (ydata<=popt[2] + popt[3]*bigbox))]
        yi = np.arange(img.shape[Waxis])[ydata2]

        #-- fit gaussian w/i each window
        if Saxis is 1:
            zi = np.sum(img_sm[ydata2, xbins[i]:xbins[i+1]], axis=Saxis)
        else:
            zi = img_sm[xbins[i]:xbins[i+1], ydata2].sum(axis=Saxis)
    
        #-- use previous iteration as starting guess
        pguess = popt
        popt,pcov = curve_fit(_gaus, yi[np.isfinite(zi)], zi[np.isfinite(zi)], p0=pguess)

        #-- plot central bin, aperture presets, gaussian fit
        if display_all:
            plt.figure()
            plt.plot(yi[np.isfinite(zi)], zi)
            plt.plot(yi[np.isfinite(zi)], _gaus(yi[np.isfinite(zi)], *popt))
            plt.axvline(pguess[2], c='tab:orange', ls='--')
            plt.axhline(pguess[1], c='tab:gray', ls=':')
            plt.show()

        # if gaussian fits off chip, then use chip-integrated answer
        if (popt[2] <= min(ydata2)+2) or (popt[2] >= max(ydata2)-2):
            ybins[i] = popt_0[2]
            #popt = popt_0
        else:
            ybins[i] = popt[2]

    popt = popt_0

    #-- from central bin loop toward decreasing index
    for i in range(int(len(xbins)/2), -1, -1):
        #print("bin #", i) 
        #-- only allow data within a box around this peak
        #print(popt[2], popt[3], popt[2] - popt[3]*bigbox, popt[2] + popt[3]*bigbox)
        ydata2 = ydata[np.where((ydata>=popt[2] - popt[3]*bigbox) &
                                (ydata<=popt[2] + popt[3]*bigbox))]
        yi = np.arange(img.shape[Waxis])[ydata2]

        #-- fit gaussian w/i each window
        if Saxis is 1:
            zi = np.sum(img_sm[ydata2, xbins[i]:xbins[i+1]], axis=Saxis)
        else:
            zi = img_sm[xbins[i]:xbins[i+1], ydata2].sum(axis=Saxis)
    
        #-- use previous iteration as starting guess
        pguess = popt
        popt,pcov = curve_fit(_gaus, yi[np.isfinite(zi)], zi[np.isfinite(zi)], p0=pguess)

        #-- plot central bin, aperture presets, gaussian fit
        if display_all:
            plt.figure()
            plt.plot(yi[np.isfinite(zi)], zi)
            plt.plot(yi[np.isfinite(zi)], _gaus(yi[np.isfinite(zi)], *popt))
            plt.axvline(pguess[2], c='tab:orange', ls='--')
            plt.axhline(pguess[1], c='tab:gray', ls=':')
            plt.show()

        # if gaussian fits off chip, then use chip-integrated answer
        if (popt[2] <= min(ydata2)+2) or (popt[2] >= max(ydata2)-2):
            ybins[i] = popt_0[2]
            #popt = popt_0
        else:
            ybins[i] = popt[2]

    popt = popt_0

        # update the box it can search over, in case a big bend in the order
        # ydata2 = ydata[np.where((ydata>= popt[2] - popt[3]*bigbox) &
        #                         (ydata<= popt[2] + popt[3]*bigbox))]

    # recenter the bin positions, trim the unused bin off in Y
    mxbins = (xbins[:-1]+xbins[1:]) / 2.
    mybins = ybins[:-1]

    # fit the order positions w/ a polynomial
    coeff_trace = np.polyfit(mxbins, mybins, ofit)
    xdata = np.arange(img.shape[Saxis])
    fit_trace = np.poly1d(coeff_trace)(xdata)
    fit_points = np.poly1d(coeff_trace)(mxbins)
    rms = np.sqrt(np.mean((fit_points - mybins)**2))
    print(">> Fit to aperture trace RMS deviation = %.3f px" % rms)

    if display is True:
        plt.figure()
        plt.imshow(np.log10(img),origin='lower',aspect='auto',cmap=cm.Greys_r)
        plt.autoscale(False)
        plt.plot(mxbins, mybins, c='tab:orange', ls='', marker='x')
        plt.plot(xdata, fit_trace, c='tab:orange', ls='--')
        plt.show()


    print(">> Trace gaussian width = "+str(popt_0[3])+' pixels')
    return fit_trace


def ap_extract(img, trace, ap_l=8, ap_u=8, skysep_l=3, skysep_u=3, sky_l=7, sky_u=7, skydeg=0,
               coaddN=1):
    """
    1. Extract the spectrum using the trace. Simply add up all the flux
    around the aperture within a specified +/- width.

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An major simplification at present. To be changed!

    2. Fits a polynomial to the sky at each column

    Note: implicitly assumes wavelength axis is perfectly vertical within
    the trace. An important simplification.

    3. Computes the uncertainty in each pixel

    Parameters
    ----------
    img : 2d numpy array
        This is the image, stored as a normal numpy array. Can be read in
        using astropy.io.fits like so:

        >>> hdu = fits.open('file.fits') # doctest: +SKIP
        >>> img = hdu[0].data # doctest: +SKIP

    trace : 1-d array
        The spatial positions (Y axis) corresponding to the center of the
        trace for every wavelength (X axis), as returned from ap_trace
    ap_l : int, optional
        The width along the Y axis on the lower side of the trace to extract.
        Note: a fixed width is used along the whole trace.
        (default is 8 pixels)
    ap_u : int, optional
        The width along the Y axis on upper side of the trace to extract.
        Note: a fixed width is used along the whole trace.
        (default is 8 pixels)
    skysep_l : int, optional
        The separation in pixels from the aperture to the lower sky window.
        (Default is 3)
    skysep_u : int, optional
        The separation in pixels from the aperture to the upper sky window.
        (Default is 3)
    sky_l : int, optional
        The width in pixels of the sky windows on the lower side of the
        aperture. (Default is 7)
    sky_u : int, optional
        The width in pixels of the sky windows on the upper side of the
        aperture. (Default is 7)
    skydeg : int, optional
        The polynomial order to fit between the sky windows.
        (Default is 0)

    Returns
    -------
    onedspec : 1-d array
        The summed flux at each column about the trace. Note: is not
        sky subtracted!
    skysubflux : 1-d array
        The integrated sky values along each column, suitable for
        subtracting from the output of ap_extract
    fluxerr : 1-d array
        the uncertainties of the flux values
    """

    onedspec = np.zeros_like(trace)
    skysubflux = np.zeros_like(trace)
    fluxerr = np.zeros_like(trace)

    for i in range(0,len(trace)):
        #-- first do the aperture flux
        # juuuust in case the trace gets too close to the edge
        widthup = ap_u
        widthdn = ap_l
        if (trace[i]+widthup > img.shape[0]):
            widthup = img.shape[0]-trace[i] - 1
        if (trace[i]-widthdn < 0):
            widthdn = trace[i] - 1

        # simply add up the total flux around the trace +/- width
        onedspec[i] = img[int(trace[i]-widthdn):int(trace[i]+widthup+1), i].sum()

        #-- now do the sky fit
        itrace = int(trace[i])
        y = np.append(np.arange(itrace-ap_l-skysep_l-sky_l, itrace-ap_l-skysep_l),
                      np.arange(itrace+ap_u+skysep_u+1, itrace+ap_u+skysep_u+sky_u+1))

        z = img[y,i]
        if (skydeg>0):
            # fit a polynomial to the sky in this column
            pfit = np.polyfit(y,z,skydeg)
            # define the aperture in this column
            ap = np.arange(trace[i]-ap_l, trace[i]+ap_u+1)
            # evaluate the polynomial across the aperture, and sum
            skysubflux[i] = np.sum(np.polyval(pfit, ap))
        elif (skydeg==0):
            skysubflux[i] = np.nanmean(z)*(ap_l + ap_u + 1)

        #-- finally, compute the error in this pixel
        sigB = np.std(z) # stddev in the background data
        N_B = len(y) # number of bkgd pixels
        N_A = ap_l + ap_u + 1 # number of aperture pixels

        # based on aperture phot err description by F. Masci, Caltech:
        # http://wise2.ipac.caltech.edu/staff/fmasci/ApPhotUncert.pdf
        fluxerr[i] = np.sqrt(np.sum((onedspec[i]-skysubflux[i])/coaddN) +
                             (N_A + N_A**2. / N_B) * (sigB**2.))

    return onedspec, skysubflux, fluxerr

