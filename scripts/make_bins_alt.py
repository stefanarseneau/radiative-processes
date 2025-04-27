import argparse
import corv
import numpy as np
import pandas as pd

from astropy.table import Table
from pathlib import Path
from tqdm import tqdm
from typing import Tuple

dir = Path(__file__).parents[1]
corvmodel = corv.models.Spectrum(model='3d_da_lte_h2', units='flam', wavl_range=(840, 1320))

def read_nlte_spectrum(source_id: str) -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """given a source, return its parameters and spectrum"""

    spec = Table.read(dir / f'data/hst_cos/{source_id.replace(' ', '-')}_coadd_FUVM_final_lpALL.fits.gz').to_pandas()
    params = pd.read_csv(dir / 'data/hst_cos/cos_params_updated.csv').query('NAMES == @source_id')
    return (params.RV.iloc[0], params.teff.iloc[0], params.logg.iloc[0]), spec.WAVE.values, spec.FLUX.values, 1 / (1e-6 + spec.ERROR)**2

def read_model_spectrum(source_id: str) -> Tuple[Tuple[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """read a spectrum and return its parameters and model spectrum"""

    params = pd.read_csv(dir / 'data/hst_cos/cos_params_updated.csv').query('NAMES == @source_id')
    rv, teff, logg = params.RV.iloc[0], params.teff.iloc[0], params.logg.iloc[0]
    wl, flux, ivar = corvmodel.wavl, corvmodel.model_spec((teff, logg)), 1e4*np.ones(corvmodel.wavl.shape[0])
    flux = corv.utils.doppler_shift(wl, flux, rv)
    return (rv, teff, logg), wl, flux, ivar

def cutout_line(wavl: np.array, flux: np.array, ivar: np.array, line: str = 'a', window: float = 100.0) -> Tuple[np.ndarray]:
    """isolate some absorption line and remove obviously bad points"""

    wavelengths = {'Lya': 1215.674, 'LyB': 1025.722, 'Lyg': 972.537, 'Lyd': 949.74287}
    wavl_cutout, flux_cutout, ivar_cutout = corv.utils.cont_norm_line(wavl, flux, ivar, wavelengths[line], window, 10)
    mask = (0 < flux_cutout) * (flux_cutout < 1.3)
    return wavl_cutout[mask], flux_cutout[mask], ivar_cutout[mask]

def get_bins(wavl: np.ndarray, flux: np.ndarray, ivar: np.ndarray, samples: int = 10, num_bins: int = 20) -> Tuple[np.ndarray]:
    """use monte carlo sampling to find the bins, centers, and uncertainties
    samples : (int) number of monte carlo samples to draw
    num_bins : (int) number of bins to cut the line into"""

    fluxes = np.array([np.random.normal(loc=y, scale=sigma, size=samples) for y, sigma in zip(flux, np.sqrt(1/ivar))])
    bin_edges = np.linspace(fluxes.min(), fluxes.max(), num_bins + 1)
    center, e_center = [], []
    for j in range(num_bins):
        bincenters = []
        for i in range(samples):
            samp_flux = fluxes[:,i]
            mask = (bin_edges[j] < samp_flux) * (samp_flux < bin_edges[j+1])
            bincenters.append(np.mean(wavl[mask]))
        center.append(np.mean(bincenters))
        e_center.append(np.std(bincenters))
    return bin_edges, np.array(center), np.array(e_center)

def get_binfile(obsnames: np.ndarray, samples: int = 10, num_bins: int = 20, spectrum_function=read_nlte_spectrum) -> np.ndarray:
    """return a numpy ndarray with all the center curves to save"""

    lines = ['Lya', 'LyB', 'Lyg', 'Lyd']
    return_obj, succeeded = [], []
    for i, obsname in tqdm(enumerate(obsnames), total=obsnames.shape[0]):
        try:
            obs_obj = []
            # read in the spectrum
            _, wavl, flux, ivar = spectrum_function(obsname)
            for j, line in enumerate(lines):
                # cut out the 100 angstrom around a given absorption line
                wavl_cutout, flux_cutout, ivar_cutout = cutout_line(wavl, flux, ivar, line = line, window=100)
                # compute the normalized flux bins, their wavelength center, and the x axis uncertainty
                bin_edges, centers, e_centers = get_bins(wavl_cutout, flux_cutout, ivar_cutout, samples, num_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                obs_obj.append([centers, e_centers, bin_centers])
            return_obj.append(obs_obj)
            succeeded.append(obsname)
        except Exception as e:
            print(f'Failed to compute {obsname} : {e}')
    return np.array(succeeded), np.array(return_obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int, default=-1)
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--n_bins', type=int, default=20)
    parser.add_argument('--array_savefile', type=str, default='linecenters.npy')
    parser.add_argument('--names_savefile', type=str, default='names_linecenters.npy')
    parser.add_argument('--use_data', type=bool, default=False)
    args = parser.parse_args()

    read_func = read_nlte_spectrum if args.use_data else read_model_spectrum
    obsnames = pd.read_csv('data/hst_cos/cos_params_updated.csv').dropna().NAMES.values[args.start:args.stop]
    names, centers = get_binfile(obsnames, args.n_samples, args.n_bins, read_func)
    np.save(args.array_savefile, centers)
    np.save(args.names_savefile, names)
