# Medley

Medley simulates 3D electron diffraction (3DED / MicroED) data. It takes the
experiment geometry from a DIALS `.expt` file (or generates a synthetic
experiment), takes the atoms from a CIF file, computes dynamical diffraction
with the Bloch-wave method (via [abTEM](https://github.com/abTEM/abTEM)), and
writes detector images and reflection tables that can be compared
with, or processed like, real data.

## Installation

```bash
git clone git@github.com:mpks/medley.git
cd medley
pip install -e .
```

This installs the `medley` and `npz_to_cbf` commands and the dependencies
(numpy, scipy, matplotlib, ase, abTEM 1.0.5, fabio).

## Quick start

### Simulate a real experiment

Use the geometry of an experiment processed with DIALS (detector,
beam, goniometer, rotation range and crystal orientation) and the 
atoms from a CIF:

```bash
medley experiment structure.cif indexed.expt -o sim_out --tag run1
```

The CIF and the `.expt` must describe the same unit cell in
the same setting (same choice of a, b, c). Medley keeps the
fractional atom coordinates from the CIF and uses the cell and
orientation from DIALS.

### Simulate a synthetic experiment

Generate an experiment from a random crystal orientation, with the detector
distance chosen to reach a given resolution at the detector corners:

```bash
medley synthetic structure.cif --d_min 0.8 --n_images 60 --orientation_seed 3
```

### Simulate only some images

Image indices start at 0 (DIALS image 1 is Medley image 0):

```bash
medley experiment structure.cif indexed.expt --images 0-9,100-109
```

### Use several CPU cores

```bash
medley experiment structure.cif indexed.expt --n_workers 8
```

Run `medley experiment --help` or `medley synthetic --help` for
the full list of options with their defaults.

## Output

All files go to the output directory (`-o`, default `sim_out`). `METHOD` is
the integration method (`vector` or `raster`), `TAG` is the run label
(`--tag`) and `NNNN` is the 4-digit image index.

| File | Contents |
|---|---|
| `image_METHOD_TAG_NNNN.npz` | Reflection table and image data for one image (see below). |
| `image_METHOD_TAG_NNNN.cbf` | Detector image in miniCBF format, readable by DIALS. Skip with `--no_cbf`. |
| `plot_METHOD_TAG_NNNN.png`, `raster_METHOD_TAG_NNNN.png` | Diagnostic plots. Skip with `--no_plots`. |
| `rocking_TAG.npz` | Rocking curves for the reflections listed in `--rocking_hkl_file`. |
| `matrices_TAG_NNNN.npz` | Bloch-wave structure matrix A and scattering matrix S(t) for one sub-angle, written with `--export_SA`. |

Each per-image NPZ contains:

- `hkl_indices`, `intensities`, `px`, `py`: Miller indices, integrated intensities and spot centroids (pixels);
- `signal`: the integrated noise-free image;
- `sg`, `sg_geom`: excitation errors at the image centre angle (omit with `--no_sg`);
- the image index and angles, plus the detector and beam geometry needed to rebuild the CBF.

```python
import numpy as np
d = np.load("sim_out/image_vector_run1_0000.npz")
print(d["hkl_indices"][:5], d["intensities"][:5])
```

## Main options

### Simulation

| Option | Default | Meaning |
|---|---|---|
| `--k_max` | 2.0 | Maximum scattering vector included (Å⁻¹); sets the resolution, d_min = 1/k_max. |
| `--sg_max` | 0.05 | Maximum excitation error for beams included in the Bloch-wave calculation (Å⁻¹). |
| `--base_thickness_A` | 800 | Crystal thickness along the beam at zero rotation (Å). At angle φ the thickness is this value divided by cos φ. |
| `--n_substeps` | 10 | Sub-angles simulated per image; their results are summed into the image. |
| `--integration` | `vector` | How sub-angles are combined: `vector` (spot list) or `raster` (pixel image). |
| `--num_phonon_configs`, `--phonon_sigma`, `--phonon_seed` | 1, 0.0, 42 | Frozen-phonon averaging: number of configurations, RMS displacement (Å) and random seed. |
| `--intensity_cut` | 0.0 | Drop reflections whose integrated intensity is at or below this value. |
| `--n_workers` | 1 | Number of worker processes. |
| `--images` | all | Images to simulate: `5`, `0-9`, `0,5,10` or `0-9,20-29`. |
| `--rocking_hkl_file` | none | File of Miller indices (HKL or 3-column) to record rocking curves for. |

### Image rendering (CBF only)

These options only affect how the CBF is rendered from the simulated signal.
They can be changed later without re-simulating (see `npz_to_cbf` below).

| Option | Default | Meaning |
|---|---|---|
| `--scale` | 10000 | Scale factor applied to the integrated signal. |
| `--psf_sigma` | 1.0 | Width (pixels) of the blur used for the spot noise. |
| `--spot_percent` | 0.001 | Spot noise as a fraction of the spot intensity (0.001 = 0.1 %). |
| `--bg_intensity`, `--bg_seed` | 1.0, 0 | Mean and seed of the Poisson background. |
| `--beam_intensity`, `--beam_sigma`, `--beam_seed` | 0, 30, 0 | Direct-beam spot: peak counts (0 = off), width (pixels), seed. |

### Synthetic mode only

| Option | Default | Meaning |
|---|---|---|
| `--wavelength_A` | 0.02851 | Electron wavelength (Å); 0.02851 Å corresponds to 160 kV. |
| `--npx`, `--npy`, `--pixel_size_mm` | 512, 512, 0.075 | Detector size and pixel size. |
| `--start_deg`, `--delta_deg`, `--n_images` | −30, 0.5, 60 | Rotation scan: start angle, width per image, number of images. |
| `--orientation_seed` | 0 | Seed for the random crystal orientation. |
| `--distance_mm` / `--d_min` / `--g_max` | 578.3 mm | Give at most one: the detector distance directly, or the resolution (Å) or scattering vector (Å⁻¹) to reach at the detector corners. |

## Re-rendering CBF images without re-simulating

The per-image NPZ files store the noise-free signal, so the CBF
images can be rebuilt with different brightness, noise or
background:

```bash
npz_to_cbf sim_out/image_vector_run1_*.npz -o rerendered --scale 100000 --beam_intensity 200
```

`npz_to_cbf` accepts the same rendering options as above, plus `--flip_y`,
which reverses the image rows. It is only needed for detector formats that
store images upside down.


## Licence

BSD 3-Clause. See [LICENSE](LICENSE).
