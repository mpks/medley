"""
plotting.py — diagnostic plots of integrated diffraction images.

Two styles are provided:

  * plot_image          — spots drawn as markers at their
                          intensity-weighted centroids on a dark
                          background (no raster image);
  * plot_detector_raster — the rasterised detector image (the
                          pixels that go into the CBF) shown on a
                          log scale, with labels overlaid.

Both label spots tiered by intensity rank:

  * top 10 %        : red circle and red h,k,l label
  * next 30 %       : grey/red h,k,l label, no circle
  * bottom 60 %     : grey dot (markers style) or none (raster)

The 40 % that are labelled give enough reference points to read
off the Miller indices of the unlabelled spots by eye.
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt

from dynamic.simulation.export_cbf import render_image


def _rank_thresholds(n, red_frac=0.10, grey_frac=0.30):
    """
    Return (n_red, n_labelled) counts from the fractions.

    n_red       : number of red-circled, red-labelled spots
    n_labelled  : number with any label (red + grey)
    """
    n_red = int(round(n * red_frac))
    n_labelled = int(round(n * (red_frac + grey_frac)))
    n_red = max(n_red, 1) if n > 0 else 0
    n_labelled = max(n_labelled, n_red)
    return n_red, n_labelled


def plot_image(image_result, detector, out_path,
               red_frac=0.10, grey_frac=0.30,
               title=None):
    """
    Plot one ImageResult and save to out_path.

    Parameters
    ----------
    image_result : ImageResult
        Provides px, py, intensities, millers.
    detector : Detector
        Provides npx, npy for the axes limits.
    red_frac : float
        Fraction (by intensity rank) circled and labelled red.
    grey_frac : float
        Additional fraction labelled grey.
    title : str or None
        Plot title; a default is built if None.
    """
    px = image_result.px
    py = image_result.py
    intensities = image_result.intensities
    millers = image_result.millers
    n = len(intensities)

    npx = detector.npx
    npy = detector.npy

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, npx)
    ax.set_ylim(npy, 0)          # detector y increases downward
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_facecolor("#1a1a1a")
    fig.patch.set_facecolor("#1a1a1a")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    if title is None:
        idx = image_result.image_index
        centre = image_result.angle_centre_deg
        title = (
            f"Image {idx}  "
            f"(centre {centre:.3f} deg, {n} spots)"
        )
    ax.set_title(title, color="white")

    if n == 0:
        plt.tight_layout()
        plt.savefig(out_path, dpi=200,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Image plot saved: {out_path}")
        return out_path

    # Rank spots by intensity, strongest first
    order = np.argsort(intensities)[::-1]
    n_red, n_labelled = _rank_thresholds(
        n, red_frac, grey_frac
    )

    for rank, i in enumerate(order):
        x = px[i]
        y = py[i]
        h, k, l = millers[i]                 # noqa: E741
        label = f"{h},{k},{l}"

        if rank < n_red:
            ax.plot(x, y, "o", color="red",
                    markersize=6,
                    markerfacecolor="none",
                    markeredgewidth=1.2,
                    clip_on=True)
            ax.text(x + 4, y - 4, label,
                    color="red", fontsize=5,
                    va="bottom", ha="left",
                    clip_on=True)
        elif rank < n_labelled:
            ax.plot(x, y, ".", color="#888888",
                    markersize=2, clip_on=True)
            ax.text(x + 3, y - 3, label,
                    color="#bbbbbb", fontsize=4,
                    va="bottom", ha="left",
                    clip_on=True)
        else:
            ax.plot(x, y, ".", color="#666666",
                    markersize=1, clip_on=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Image plot saved: {out_path}")
    return out_path


def plot_filename(out_dir, method, tag, image_index):
    """Per-image labels-plot path, with method and 4-digit
    zero-padded index."""
    name = f"plot_{method}_{tag}_{image_index:04d}.png"
    return os.path.join(out_dir, name)


def raster_filename(out_dir, method, tag, image_index):
    """Per-image raster-plot path, with method and 4-digit
    zero-padded index."""
    name = f"raster_{method}_{tag}_{image_index:04d}.png"
    return os.path.join(out_dir, name)


def plot_detector_raster(image_result, detector, params,
                         out_path, red_frac=0.10,
                         grey_frac=0.30, title=None):
    """
    Plot the rasterised detector image (the actual pixels that
    go into the CBF) with Miller-index labels overlaid.

    The image is the same rasterisation used for the CBF
    (spots placed at centroids, PSF-broadened, scaled, noisy),
    shown on a log intensity scale with a binary colormap so
    weak spots remain visible.  Labels follow the same tiering
    as plot_image: red top 10 %, grey next 30 %.

    Parameters
    ----------
    params : CbfParams
        Controls the rasterisation (scale, psf_sigma, noise).
    """
    npx = detector.npx
    npy = detector.npy

    image = render_image(
        image_result.signal, detector.beam_centre_px,
        params, image_result.image_index,
    )
    display = np.log1p(image.astype(np.float64))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(
        display,
        cmap="binary",
        origin="upper",
        extent=[0, npx, npy, 0],
        interpolation="nearest",
        aspect="equal",
    )

    px = image_result.px
    py = image_result.py
    intensities = image_result.intensities
    millers = image_result.millers
    n = len(intensities)

    if title is None:
        idx = image_result.image_index
        centre = image_result.angle_centre_deg
        title = (
            f"Image {idx}  "
            f"(centre {centre:.3f} deg, {n} spots)"
        )
    ax.set_title(title)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.set_xlim(0, npx)
    ax.set_ylim(npy, 0)

    if n > 0:
        order = np.argsort(intensities)[::-1]
        n_red, n_labelled = _rank_thresholds(
            n, red_frac, grey_frac
        )
        for rank, i in enumerate(order):
            x = px[i]
            y = py[i]
            if not (0 <= x < npx and 0 <= y < npy):
                continue
            h, k, l = millers[i]             # noqa: E741
            label = f"{h},{k},{l}"
            if rank < n_red:
                ax.plot(x, y, "o", color="red",
                        markersize=5,
                        markerfacecolor="none",
                        markeredgewidth=1.0,
                        clip_on=True)
                ax.text(x + 3, y - 3, label,
                        color="red", fontsize=4,
                        va="bottom", ha="left",
                        clip_on=True)
            elif rank < n_labelled:
                ax.text(x + 3, y - 3, label,
                        color="#d04040", fontsize=3.5,
                        va="bottom", ha="left",
                        alpha=0.8, clip_on=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Raster plot saved: {out_path}")
    return out_path


def plot_images(images, detector, out_dir, tag,
                red_frac=0.10, grey_frac=0.30):
    """Plot every ImageResult (labels style); return paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for image in images:
        path = plot_filename(out_dir, image.method, tag,
                             image.image_index)
        plot_image(image, detector, path,
                   red_frac=red_frac,
                   grey_frac=grey_frac)
        paths.append(path)
    return paths


def plot_images_raster(images, detector, params, out_dir,
                       tag, red_frac=0.10, grey_frac=0.30):
    """Plot every ImageResult (raster style); return paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for image in images:
        path = raster_filename(out_dir, image.method, tag,
                              image.image_index)
        plot_detector_raster(image, detector, params, path,
                             red_frac=red_frac,
                             grey_frac=grey_frac)
        paths.append(path)
    return paths
