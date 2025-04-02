## Radiative Processes Project
---

To run the notebook, you'll need to install the python package `corv`:

`pip install git+https://github.com/vedantchandra/corv`

To make the black curves from the notebook, run the following command:

`python3 scripts/make_bins.py --start_idx=0 --stop_idx=-1 --n_samples=10 --n_bins=20 --savefile=linecenters.npy`

which will return a numpy array of shape `(n,4,3,m)` where `n` indexes the spectrum, the `4` dimension indexes the absorption line (`[a,b,g,d]`), `3` indexes the column to use (`[bincenter, e_bincenter, normalized_flux]`), and `m` is the `n_bins` parameter.

---

**Papers Related To Line Shapes Generally:**

* https://ui.adsabs.harvard.edu/abs/2022ApJ...927...70C/abstract

**UV-Specific Papers**

* https://ui.adsabs.harvard.edu/abs/2023MNRAS.526.5800S/abstract

---

The `data/` directory contains `reference_objs.csv`, which are parameters for all white dwarfs in the SPY sample. 

`data/tlusty_atmospheres/{teff}.{logg}_{model}/t{teff}g{logg*100}.spec` contains the following model spectra for teffs of 17,000 and 20,000 and loggs of 7.0, 8.0, and 9.0:
* `tre` : line shapes from Tremblay et al 2013,2015
* `vcs` : cannonical Vidal Cooper Smith line shapes
* `occprob_HM88` : outer wings of absorption lines are corrected by an occupation probability
* `idyn` : line shapes calculated with a full treatment of ion dynamics beyond dipole coulomb interaction
* `allon` : `occprob_HM88` and `idyn` physics combined into a single line shape.

`data/sp_fits/*.fits.gz` are the observed spectra of the SPY survey in the heliocentric reference frame.

**Probably Just Ignore These**

`radial-velocities/nlte_core/{lines}_{window}aa.csv` are radial velocities measured using the NLTE core of some combination of one or multiple absorption lines. These radial velocities are the most reliable.

`radial-velocities/lte_smooth/{model}/{lines}_window_{n}.csv` contains radial velocities measured using either the Tremblay `3d_da_lte_h2` model spectra or analytic Voigt profiles on a spectrum that has been convolved down to spectral resolution of R=1800 so that there is no information about the NLTE core.
