#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
iraf_um.py: utility functions for CCD data reduction and analysis

27 Sep 2021: Created w/ IRAF-inspired imstat 
28 Sep 2021: Add function inv_median
30 Jul 2022: Add optional argument 'fields' to imstat
"""

__author__ = "Julien Morin <julien.morin@umontpellier.fr>"
__date__ = "2022-07-30"
__version__ = "0.3"

from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sstats

from astropy.nddata import CCDData
from astropy.stats import mad_std
from astropy.stats import sigma_clipped_stats
from astropy.table import Table

def imstat(frame_id, ccd_data, sig_clip=None, fields=('frame id', 'npix', 'min',
    'max', 'mean', 'median', 'mode', 'std', 'mad', 'unit'), fmt='8.3f', verbose=True, return_table=False):
    """
    Compute statistics similarly to the IRAF imstat function for an image or a 
    series of images, and store in an astropy.table.Table object.

    Parameters
    ----------
    frame_id: str, list of str
        frame identification to display as 1st column.

    ccd_data: astropy.nddata.CCDData object or list of thereof. Number of 
        elements must be identical to frame_id.

    sig_clip: float or None, optional [None]
        if a float is provided the mean, median and standard deviation are
        computed using astropy.stats.sigma_clipped_stats passing sigma=sig_clip.

    fields: tuple(str)
        fields to compute and print

    fmt: str ['8.3f']
        format string for display in the result astropy.table.Table object.

    verbose: bool [True]
        if True print the output table.

    return_table: bool [False]
        if True return the output table.
    """
    if type(frame_id) != list:
        frame_id = [frame_id]
        ccd_data = [ccd_data]

    # create OrderedDict to w/ empty lists to store the statistics of images
    stats = OrderedDict()
    for f in fields:
        stats[f] = []
    if 'frame id' in fields:
        stats['frame id'] = frame_id
    for ccd in ccd_data:
        if 'npix' in fields:
            stats['npix'].append(ccd.shape[0]*ccd.shape[1])
        if 'min' in fields:
            stats['min'].append(np.min(ccd))
        if 'max' in fields:
            stats['max'].append(np.max(ccd))
        if sig_clip is None:
            if 'mean' in fields:
                mea = np.mean(ccd)
            if 'median' in fields:
                med = np.median(ccd)
            if 'std' in fields:
                std = np.std(ccd)
        else:
            mea, med, std = sigma_clipped_stats(ccd, sigma=sigclip)
        if 'mean' in fields:
            stats['mean'].append(mea)
        if 'median' in fields:
            stats['median'].append(med)
        if 'mode' in fields:
            stats['mode'].append(sstats.mode(ccd)[0][0][0])
        if 'std' in fields:
            stats['std'].append(std)
        if 'mad' in fields:
            stats['mad'].append(mad_std(ccd))
        if 'unit' in fields:
            try:
                stats['unit'].append(str(ccd.unit))
            except:
                stats['unit'].append(None)

    stat_names = list(stats.keys())
    stat_fields = list(stats.values())
    stat_tab = Table(stat_fields,  names=stat_names)
    for f in ['min', 'max', 'mean', 'median', 'mode', 'std', 'mad']: 
        if f in fields:
            stat_tab[f].info.format = fmt
    if 'npix' in fields:
        stat_tab['npix'].info.format = '9d'
    #
    if verbose:
        print(stat_tab)
    #
    if return_table:
        return stat_tab

def inv_median(a):
    """
    Inverse median function to be used e.g. for scaling in ccdproc.combine.
    """
    return 1./np.median(a)
