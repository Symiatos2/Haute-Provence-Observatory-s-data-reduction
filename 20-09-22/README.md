# HAP905P - Atelier Astrophysique Observationnelle

Ressources pour l'analyse de données astronomiques dans le cadre de l'Atelier Astrophysique Observationnelle (HAP905P) à l'Observatoire de Haute-Provence des Masters 2 Astrophysique et Cosmos Champs Particules.

Ressources for astronomical data analysis in the framework of the Observational Astrophysics Workshop (HAP905P) at Observatoire de Haute-Provence of Masters 2 Astrophysique and Cosmos Champs Particules.

## Description
A series of Jupyter notebooks with example data are provided for spectroscopy and photometry, along with python libraries and scripts.

## Pre-requisite
The following python libraries are used, they can be installed via PIP or Anaconda:
 * Numpy
 * Scipy
 * Matplotlib
 * Astropy
 * astroplan
 * ccdproc
 * Astroalign
 * Photutils
 * Specutils

## Usage
Start the jupyter-notebook server from the chosen working directory (e.g. `spectroscopy`). Python needs to access the library `iraf_um.py` it can be either added to the `PYTHONPATH`, or copied or symlinked to the working directory.

## Authors and acknowledgment
Notebooks, libraries and scripts written by Julien Morin based on:
 * the online documentation of Astropy and affiliated packages
 * reimplementation of IRAF functionalities
 * adaption of the PyDIS software by J.R.A Davenport https://github.com/StellarCartography/pydis

## License
The project `HAP905P - Atelier Astrophysique Observationnelle` is licensed under the GNU GPL v3.0 https://www.gnu.org/licenses/gpl-3.0.en.html.

## Project status
Gitlab repository created on 28 July 2022, active development.
