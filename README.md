# Medley

Medley simulates 3D electron diffraction (3DED / MicroED) data. It takes the
experiment geometry from a DIALS `.expt` file (or generates a synthetic
experiment) and the atoms from a CIF, computes dynamical diffraction with the
Bloch-wave method of [abTEM](https://github.com/abTEM/abTEM), and writes
per-image reflection tables and images that can be compared with, or
processed like, real data.

## Installation

```bash
git clone git@github.com:mpks/medley.git
cd medley
pip install -e .
```

Requires numpy < 2, scipy < 1.17, matplotlib, ase, abTEM 1.0.5 and fabio
(installed automatically).

## Workflow

A run is described by a JSON parameter file that holds every option.

```bash
medley params experiment -o params.json      # 1. write a file with all defaults
#                                              2. edit it: set cif_file, expt_file, ...
medley optimise params.json                  # 3. optional: choose n_substeps, k_max, sg_max
medley run params_optimised.json             # 4. simulate
```

Options given after the file override it, for example
`medley run params.json --images 0-9 --n_workers 8`. Each run saves the
parameters it used to `<out_dir>/medley_params.json`, which repeats the run
with `medley run`.

Use `medley params synthetic` to start a synthetic experiment instead: a random crystal
orientation, and a detector distance set by the resolution wanted at the
detector corners. The same options can also be given directly on the command line
(`medley experiment structure.cif indexed.expt ...`). Run
`medley experiment --help` for the full list.

The CIF and the `.expt` must describe the same unit cell in the same setting.
Medley takes the fractional atom coordinates from the CIF, and the cell and
orientation from DIALS.

### Main parameters

| Parameter | Meaning |
|---|---|
| `k_max` | Beams are included out to \|g\| = k_max / 2 (Å⁻¹), so spots reach d = 2 / k_max. |
| `sg_max` | Largest excitation error of a beam included in the calculation (Å⁻¹). |
| `base_thickness_A` | Crystal thickness at zero tilt (Å); at tilt φ it is this value / cos φ. A list of thicknesses costs little more than one, and all are stored together. |
| `n_substeps` | Orientations simulated per image and summed into it. |
| `images` | Images to simulate: `5`, `0-9`, `0,5,10`, `0-9,20-29`. Index 0 is DIALS image 1. |
| `n_workers`, `threads_per_worker` | Worker processes, and the threads each may use. Keep their product within the cores available. |
| `experiment` | Which experiment of the `.expt` file to simulate (see below). |
| `cbf`, `plots` | Also write CBF images or diagnostic plots during the run (first thickness only). Off by default; CBFs can be made later from the NPZ files. |

### Choosing the settings

`medley optimise params.json` picks `n_substeps`, `k_max` and `sg_max` for
the experiment and thickness in the file. It runs short simulations on a few
test images and compares their spot intensities (R1) as each setting is
refined. The result is written to `params_optimised.json`, with a report
alongside it in `params_optimised_report.json`. With several thicknesses it
tunes for the thickest. See `medley optimise --help` for the tolerance and
search ranges.


## Output

Files go to `out_dir` (default `sim_out`). `METHOD` is the integration
method (`vector` or `raster`), `TAG` the run label and `NNNN` the image
index.

| File | Contents |
|---|---|
| `image_METHOD_TAG_NNNN.npz` | One image, all thicknesses (see below). |
| `rocking_TAG.npz` | Rocking curves of the reflections in `rocking_hkl_file`, per thickness. |
| `medley_params.json` | The parameters of the run. |
| `image_METHOD_TAG_NNNN.cbf`, `plot_*.png`, `raster_*.png` | Only with `cbf` / `plots`. |
| `matrices_TAG_NNNN.npz` | Bloch-wave matrices A and S(t) for one substep, only with `export_SA`. |

Each per-image NPZ holds `thicknesses_A` (T), the reflection list
`hkl_indices` (N), and per thickness `intensities`, `px`, `py` (T × N) and
the noise-free image `signal` (T × npy × npx). It also holds the excitation
errors `sg` and `sg_geom` (skip with `no_sg`), the experiment index, and the
geometry needed to render images.

```python
import numpy as np
d = np.load("sim_out/image_vector_run1_0000.npz")
print(d["thicknesses_A"], d["hkl_indices"][:5], d["intensities"][:, :5])
```

## Tools for the NPZ files

| Command | Does |
|---|---|
| `medley.integrate` | Sums the per-image tables into a SHELX HKLF 4 file for the scan, one per thickness. |
| `medley.npz_to_nexus` | Writes the images as one NeXus/NXmx file with the `.expt` geometry, for `dials.import`. Writes one file per thickness. |
| `npz_to_cbf` | Renders CBF images (first thickness, or `--thickness`). |
| `medley.npz_to_png` | Renders PNG images for a quick look. |
| `medley.compare_hkl` | R1, wR2 and correlations between two HKLF 4 files. |
| `medley.invert_cif` | Writes the inverted structure of a CIF (the other enantiomer). |

The rendering tools take `--scale`, `--bg_intensity`, `--beam_intensity`,
`--psf_sigma`, `--spot_percent`, the seed. Brightness and
noise can therefore be changed without re-simulating. For example:

```bash
medley.integrate sim_out/image_vector_run1_*.npz -o run1.hkl --scale=100000
medley.npz_to_nexus indexed.expt sim_out/image_vector_run1_*.npz -o run1.nxs
```

## Licence

BSD 3-Clause. See [LICENSE](LICENSE).
