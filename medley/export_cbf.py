"""
export_cbf.py — render integrated images to miniCBF files.

An image is reconstructed from the integrated signal and noise
images (produced by either integrator) plus three render-time
noise contributions:

  * signal     : the integrated signal, scaled so its strongest
                 pixel is CbfParams.scale counts;
  * spot noise : the stored noise image (a faint, PSF-broadened
                 halo proportional to the spots), added as is;
  * background : a random Poisson field over the whole image,
                 generated from bg_seed combined with the image
                 index, at level bg_intensity;
  * beam       : a deterministic Gaussian blob at the direct
                 beam, generated from beam_seed combined with
                 the image index, of width beam_sigma and peak
                 beam_intensity.

Only signal and spot noise come from the simulation (stored in
the NPZ); background and beam are regenerated here, so they can
be changed by re-rendering without re-running the simulation.
The header is filled from the detector and beam geometry.
"""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


# ----------------------------------------------------------------
# CBF render parameters (three-level noise)
# ----------------------------------------------------------------
@dataclass(frozen=True)
class CbfParams:
    """
    Render-time parameters.

    scale : float
        Multiplicative factor applied to the integrated signal
        (preserves the true relative spot intensities).
    psf_sigma : float
        Gaussian sigma (pixels) of the point-spread blur used to
        build the spot-noise halo from the signal.
    spot_percent : float
        Spot-level noise as a fraction of the (blurred) signal
        (e.g. 0.001 for 0.1 %).
    bg_intensity : float
        Mean of the Poisson background field.
    bg_seed : int
        Master seed for the background; combined with the image
        index so each image gets a distinct, reproducible field.
    beam_sigma : float
        Sigma (pixels) of the direct-beam Gaussian blob.
    beam_intensity : float
        Peak counts of the direct-beam blob (0 disables it).
    beam_seed : int
        Master seed for the direct-beam blob; combined with the
        image index.
    flip_y : bool
        If True, reverse the slow (row / y) axis of the signal
        before rendering.  Undoes a y-inverted image so it
        agrees with the (unchanged) header geometry.  The beam
        blob is still placed at beam_centre_px and the header
        still reports that centre, so no beam-centre transform
        is needed.
    """
    scale: float = 10000.0
    psf_sigma: float = 1.0
    spot_percent: float = 0.001
    bg_intensity: float = 1.0
    bg_seed: int = 0
    beam_sigma: float = 30.0
    beam_intensity: float = 0.0
    beam_seed: int = 0
    flip_y: bool = False


# ----------------------------------------------------------------
# Per-image RNG (seed-seeds-the-seed)
# ----------------------------------------------------------------
def _image_rng(master_seed, image_index):
    """
    A reproducible per-image generator: the master seed and the
    image index together seed the generator, so one master seed
    fixes the whole scan image by image.
    """
    seq = np.random.SeedSequence([master_seed, image_index])
    return np.random.default_rng(seq)


# ----------------------------------------------------------------
# Noise contributions
# ----------------------------------------------------------------
def _scaled_signal(signal, scale):
    """Apply the scale factor to the signal (preserving the
    true relative intensities of the spots)."""
    return signal * scale


def _background(shape, params, image_index):
    """Random Poisson background field for this image."""
    if params.bg_intensity <= 0:
        return np.zeros(shape)
    rng = _image_rng(params.bg_seed, image_index)
    return rng.poisson(
        params.bg_intensity, size=shape
    ).astype(np.float64)


def _beam_blob(shape, beam_centre_px, params, image_index):
    """
    Diffuse-scatter blob at the direct beam: a smooth Gaussian
    envelope with Poisson shot noise on top, so it is not a
    clean Gaussian.  The per-image RNG (from beam_seed and the
    image index) makes the noise reproducible.
    """
    if params.beam_intensity <= 0 or params.beam_sigma <= 0:
        return np.zeros(shape)
    npy, npx = shape
    cx, cy = beam_centre_px
    yy, xx = np.mgrid[0:npy, 0:npx]
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    envelope = params.beam_intensity * np.exp(
        -r2 / (2.0 * params.beam_sigma ** 2)
    )
    # Poisson shot noise using the envelope as the local mean,
    # so the noise grows with the blob's local brightness and
    # vanishes far from the beam.
    rng = _image_rng(params.beam_seed, image_index)
    return rng.poisson(envelope).astype(np.float64)


def _spot_noise(scaled_signal, params):
    """
    Spot-level noise halo: the scaled signal blurred by the
    point-spread function and scaled by spot_percent.  This is
    the deterministic faint halo around every spot, produced at
    render time (not stored)."""
    if params.spot_percent <= 0 or params.psf_sigma <= 0:
        return np.zeros_like(scaled_signal)
    blurred = gaussian_filter(scaled_signal, params.psf_sigma)
    return blurred * params.spot_percent


# ----------------------------------------------------------------
# Render
# ----------------------------------------------------------------
def render_image(signal, beam_centre_px, params, image_index):
    """
    Build the final integer detector image from the sharp
    signal.

    If params.flip_y is set, the signal's slow (row / y) axis is
    reversed first, so a y-inverted signal is corrected before
    the beam blob, background, and header (all in the un-flipped
    frame) are applied.

    Layers added to the scaled signal:
      * spot noise : the scaled signal blurred (psf_sigma) and
                     scaled by spot_percent;
      * background : a random Poisson field (bg_seed + index);
      * beam       : the direct-beam blob with shot noise
                     (beam_seed + index).

    Returns
    -------
    ndarray (npy, npx), int32
    """
    if params.flip_y:
        signal = signal[::-1, :]
    shape = signal.shape
    scaled = _scaled_signal(signal, params.scale)
    if params.psf_sigma > 0:
        scaled = gaussian_filter(scaled, params.psf_sigma)
    img = scaled
    img = img + _spot_noise(scaled, params)
    img = img + _background(shape, params, image_index)
    img = img + _beam_blob(
        shape, beam_centre_px, params, image_index
    )
    img = np.clip(np.round(img), 0, None)
    return img.astype(np.int32)


# ----------------------------------------------------------------
# Header
# ----------------------------------------------------------------
def build_header(distance_mm, pixel_size_mm, beam_centre_px,
                 wavelength_A, start_angle_deg,
                 angle_increment_deg):
    """
    Build the _array_data.header_contents string in the
    ELDICO ED-1 / Eiger Quadro miniCBF format from plain
    geometry values (so it works from a loaded NPZ too).
    """
    px_m = pixel_size_mm * 1e-3
    d_m = distance_mm * 1e-3
    cx, cy = beam_centre_px
    wl = wavelength_A
    ts = datetime.datetime.utcnow().strftime(
        "%Y-%m-%dT%H:%M:%S.000"
    )
    lines = [
        "# Detector: Eiger Quadro S/N E-01-0000",
        f"# {ts}",
        f"# Pixel_size {px_m:.2e} m x {px_m:.2e} m",
        "# Silicon sensor, thickness 0.000450 m",
        "# Exposure_time 1.0000000 s",
        "# Exposure_period 1.0000001 s",
        "# Tau = 0 s",
        "# Count_cutoff 16777216 counts",
        "# Threshold_setting: 21743 eV",
        "# Gain_setting: high gain (vrf = -0.150)",
        "# N_excluded_pixels = 0",
        "# Excluded_pixels: badpix_mask.tif",
        "# Flat_field: x.tif",
        "# Trim_file: x.bin",
        "# Image_path: /data/synthetic/",
        "# Retrigger_mode: 1",
        f"# Wavelength {wl:.6f} A",
        f"# Beam_xy ({cx:.2f}, {cy:.2f}) pixels",
        f"# Detector_distance {d_m:.6f} m",
        f"# Start_angle {start_angle_deg:.3f} deg.",
        f"# Angle_increment {angle_increment_deg:.3f} deg.",
        "# Detector_2theta 0.000 deg.",
        "# Alpha 0.0000 deg.",
        "# Kappa 0.0000 deg.",
        f"# Phi {start_angle_deg:.3f} deg.",
        f"# Phi_increment {angle_increment_deg:.3f} deg.",
        "# Omega 0.0000 deg.",
        "# Omega_increment 0.0000 deg.",
        "# Chi 267.000 deg.",
        "# Oscillation_axis PHI",
    ]
    return "\r\n".join(lines) + "\r\n"


# ----------------------------------------------------------------
# Writing
# ----------------------------------------------------------------
def write_cbf(image_int32, header_contents, output_path):
    """
    Write a miniCBF using the PILATUS_1.2 convention so dxtbx
    selects the electron-probe Eiger format.
    """
    import fabio.cbfimage
    fabio_header = {
        "_array_data.header_convention": "PILATUS_1.2",
        "_array_data.header_contents": header_contents,
    }
    cbf = fabio.cbfimage.CbfImage(
        data=image_int32, header=fabio_header
    )
    cbf.write(output_path)


def cbf_filename(out_dir, method, tag, image_index):
    """Per-image CBF path including the integration method."""
    name = f"image_{method}_{tag}_{image_index:04d}.cbf"
    return os.path.join(out_dir, name)


def save_image_cbf(image_result, detector, beam, scan,
                   params, out_dir, tag):
    """
    Render one ImageResult to a CBF during a simulation run.
    Uses the in-memory ImageResult signal/noise and the
    detector/beam/scan objects for the header.
    """
    os.makedirs(out_dir, exist_ok=True)
    delta = scan.delta_deg
    start_angle = image_result.angle_centre_deg - delta / 2.0
    img = render_image(
        image_result.signal, detector.beam_centre_px,
        params, image_result.image_index,
    )
    header = build_header(
        detector.distance_mm, detector.pixel_size_mm,
        detector.beam_centre_px, beam.wavelength_A,
        start_angle, delta,
    )
    path = cbf_filename(
        out_dir, image_result.method, tag,
        image_result.image_index,
    )
    write_cbf(img, header, path)
    return path
