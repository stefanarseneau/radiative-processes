from astropy.table import Table
from typing import Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

import argparse
import corv

dirpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
corvmodel = corv.models.Spectrum(model = '3d_da_lte_h2', units = 'flam', wavl_range = (3600, 9000))

def read_nlte_spectrum(source_id : str) -> Tuple[np.array, np.array, np.array]:
    """read a spectrum and return its parameters
    """
    spec = Table.read(os.path.join(dirpath, 'data', 'sp_fits', f'{source_id}.fits.gz')).to_pandas()
    fit = Table.read(os.path.join(dirpath, 'data', 'radial_velocities', 'nlte_core', 'ab_15aa.csv')).to_pandas()
    fitrow = fit.query(f"obsname == '{source_id}'")
    return (fitrow.nlte_rv.values[0], fitrow.nlte_teff.values[0], fitrow.nlte_logg.values[0]), spec.wavl, spec.flux, spec.ivar

def read_model_spectrum(source_id : str) -> Tuple[np.array, np.array, np.array]:
    """read a spectrum and return its parameters
    """
    fit = Table.read(os.path.join(dirpath, 'data', 'radial_velocities', 'nlte_core', 'ab_15aa.csv')).to_pandas()
    fitrow = fit.query(f"obsname == '{source_id}'")
    rv, teff, logg = fitrow.nlte_rv.values[0], fitrow.nlte_teff.values[0], fitrow.nlte_logg.values[0]
    wavl, flux = corvmodel.wavl, corvmodel.model_spec((teff, logg))
    ivar = 1e4*np.ones(flux.shape[0])
    flux = corv.utils.doppler_shift(wavl, flux, rv)
    return (rv, teff, logg), wavl, flux, ivar

def cutout_line(wavl : np.array, flux : np.array, ivar : np.array, line : str = 'a', window : float = 100.0) -> Tuple[np.array, np.array, np.array]:
    """isolate the 100 angstrom around some absorption line and remove obviously bad points
    """
    wavelengths = {'a' : 6564.61, 'b' : 4862.68, 'g' : 4341.68, 'd' : 4102.89}
    wavl_cutout, flux_cutout, ivar_cutout = corv.utils.cont_norm_line(wavl, flux, ivar, wavelengths[line], window, 10)
    mask = (0 < flux_cutout) * (flux_cutout < 1.3)
    return wavl_cutout[mask], flux_cutout[mask], ivar_cutout[mask]

def get_bins(wavl : np.array, flux : np.array, ivar : np.array, samples : int = 10, num_bins : int = 20) -> Tuple[np.array, np.array]:
    """use monte carlo sampling to find the bins, centers, and uncertainties
    samples : (int) number of monte carlo samples to draw
    num_bins : (int) number of bins to cut the line into
    """
    fluxes = np.array([np.random.normal(loc=y, scale=sigma, size=samples) for y, sigma in zip(flux, np.sqrt(1/ivar))])
    bin_edges = np.linspace(np.min(fluxes), np.max(fluxes), num_bins + 1)
    center, e_center = [], []
    for j in range(num_bins):
        bincenters = []
        for i in range(samples):
            samp_flux = fluxes[:,i]
            mask = (bin_edges[j] < samp_flux) * (samp_flux < bin_edges[j+1])
            bincenters.append(np.mean(wavl[mask]))
        center.append(np.mean(bincenters))
        e_center.append(np.std(bincenters))
    return np.array(bin_edges), np.array(center), np.array(e_center)

def get_binfile(obsnames : np.array, samples : int = 10, num_bins : int = 20, spectrum_function = read_nlte_spectrum) -> np.ndarray:
    """return a numpy ndarray with all the center curves to save
    """
    lines = ['a', 'b', 'g', 'd']
    return_obj, succeeded = [], []
    for i, obsname in tqdm(enumerate(obsnames), total = obsnames.shape[0]):
        try:
            obs_obj = []
            # read in the spectrum
            params, wavl, flux, ivar = spectrum_function(obsname)
            for j, line in enumerate(lines):
                # cut out the 100 angstrom around a given absorption line
                wavl_cutout, flux_cutout, ivar_cutout = cutout_line(wavl, flux, ivar, line = line, window = 100)
                # compute the normalized flux bins, their wavelength center, and the x axis uncertainty
                bin_edges, centers, e_centers = get_bins(wavl_cutout, flux_cutout, ivar_cutout, samples = samples, num_bins = num_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                obs_obj.append([centers, e_centers, bin_centers])
            return_obj.append(obs_obj)
            succeeded.append(obsname)
        except Exception as e:
            print(f"failed to compute {obsname} : {e}")
    return np.array(succeeded), np.array(return_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--stop_idx', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_bins', type=int, default=20)
    parser.add_argument('--array_savefile', type=str, default='linecenters.npy')
    parser.add_argument('--names_savefile', type=str, default='names_linecenters.npy')
    parser.add_argument('--use_data', type=bool, default=False)
    args = parser.parse_args()

    function = read_nlte_spectrum if args.use_data else read_model_spectrum
    spyobjs = pd.read_csv(os.path.join('data', 'reference_objs.csv'))
    obsnames = spyobjs.obsname.values[args.start_idx:args.stop_idx]
    names, centers = get_binfile(obsnames, samples = args.n_samples, num_bins = args.n_bins, spectrum_function=function)
    np.save(args.array_savefile, centers)
    np.save(args.names_savefile, names)
