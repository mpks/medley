"""
calibration.py — choose the sample-to-detector distance for a
synthetic experiment from a desired resolution at the detector
corners.

The maximum resolution on a flat detector falls at the corner
furthest from the beam centre.  Assuming the beam centre is at
the centre of the detector, the furthest corner sits at radius

    r_corner = sqrt((npx/2 * px_x)^2 + (npy/2 * px_y)^2)   [mm]

A reflection landing there has scattering angle 2*theta with

    tan(2 theta) = r_corner / distance_mm

and Bragg's law gives the resolution

    d = lambda / (2 sin theta),     g = 1 / d.

The single builder distance_from_resolution accepts exactly one
of g_max (A^-1), d_min (A) or distance_mm and always returns the
distance in mm.
"""

from __future__ import annotations

import numpy as np


def corner_radius_mm(npx, npy, px_x_mm, px_y_mm):
    """
    Radius (mm) of the detector corner furthest from a beam
    centre placed at the detector centre.  Works for
    rectangular detectors and non-square pixels.
    """
    half_x_mm = (npx / 2.0) * px_x_mm
    half_y_mm = (npy / 2.0) * px_y_mm
    return float(np.hypot(half_x_mm, half_y_mm))


def resolution_to_distance(d_min_A, wavelength_A,
                           r_corner_mm):
    """
    Sample-to-detector distance (mm) such that the furthest
    corner corresponds to resolution d_min_A.

    Raises if the requested resolution is finer than the
    wavelength allows (sin theta > 1) or if the implied
    scattering angle reaches 90 degrees.
    """
    sin_theta = wavelength_A / (2.0 * d_min_A)
    if sin_theta >= 1.0:
        raise ValueError(
            f"d_min={d_min_A} A is finer than the wavelength "
            f"limit at lambda={wavelength_A} A "
            "(sin theta >= 1)."
        )
    theta = np.arcsin(sin_theta)
    two_theta = 2.0 * theta
    if two_theta >= np.pi / 2.0:
        raise ValueError(
            "Scattering angle 2 theta reaches 90 degrees; "
            "geometry is unphysical for a flat detector."
        )
    distance_mm = r_corner_mm / np.tan(two_theta)
    return float(distance_mm)


def distance_to_resolution(distance_mm, wavelength_A,
                           r_corner_mm):
    """
    Inverse of resolution_to_distance: the corner resolution
    d_min (A) produced by a given distance.
    """
    two_theta = np.arctan2(r_corner_mm, distance_mm)
    theta = two_theta / 2.0
    sin_theta = np.sin(theta)
    if sin_theta <= 0:
        raise ValueError("Non-positive scattering angle.")
    d_min_A = wavelength_A / (2.0 * sin_theta)
    return float(d_min_A)


def distance_from_resolution(wavelength_A, npx, npy,
                             px_x_mm, px_y_mm,
                             g_max=None, d_min=None,
                             distance_mm=None,
                             report=True):
    """
    Return the sample-to-detector distance in mm.

    Exactly one of the following must be supplied:
      g_max       : maximum scattering vector at the corner,
                    in A^-1  (d_min = 1 / g_max)
      d_min       : resolution at the corner, in A
      distance_mm : the distance directly (passed through)

    Parameters
    ----------
    wavelength_A : float
    npx, npy : int          detector size in pixels
    px_x_mm, px_y_mm : float pixel size along x and y (mm)
    report : bool
        If True, print the resulting corner resolution.
    """
    supplied = [
        ("g_max", g_max),
        ("d_min", d_min),
        ("distance_mm", distance_mm),
    ]
    given = [(name, v) for name, v in supplied
             if v is not None]
    if len(given) != 1:
        names = ", ".join(n for n, _ in supplied)
        raise ValueError(
            "Supply exactly one of "
            f"{names}; got {len(given)}."
        )

    r_corner = corner_radius_mm(npx, npy, px_x_mm, px_y_mm)

    mode, value = given[0]
    if mode == "distance_mm":
        dist = float(value)
    elif mode == "d_min":
        dist = resolution_to_distance(value, wavelength_A,
                                      r_corner)
    else:                       # g_max
        if value <= 0:
            raise ValueError("g_max must be positive.")
        d_min_from_g = 1.0 / value
        dist = resolution_to_distance(d_min_from_g,
                                      wavelength_A, r_corner)

    if report:
        d_corner = distance_to_resolution(dist, wavelength_A,
                                          r_corner)
        print(
            f"Detector distance: {dist:.3f} mm  "
            f"(corner resolution d_min = {d_corner:.4f} A, "
            f"g_max = {1.0 / d_corner:.4f} A^-1)"
        )

    return dist
