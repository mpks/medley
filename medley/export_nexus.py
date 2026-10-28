"""
export_nexus.py — write simulated images to a NeXus/NXmx file that
carries the full experiment geometry.

DIALS reads NXmx with its generic reader (FormatNXmx, or FormatNXmxED
when the probe is electrons), whatever the detector.  So, unlike a
miniCBF, the file does not have to match a detector that DIALS
already knows: the detector, beam, goniometer and scan are all in the
file.

Geometry is taken from one experiment of a DIALS .expt file and
converted from the dxtbx (imgCIF) frame to the NeXus (McStas) frame,
x -> -x, z -> -z, as dxtbx's own NXmx writer does.

DIALS' NXmx reader always places the beam exactly along z.  If the
experiment's beam is slightly tilted, the whole geometry (detector and
goniometer) is rotated by the small rotation that brings the beam
onto z, so every relative angle -- and so every spot position on the
detector -- is preserved.  The file's lab frame then differs from the
.expt's by that small rotation (printed when it is applied).
"""

from __future__ import annotations

import datetime
import json

import h5py
import numpy as np

IMGCIF_TO_MCSTAS = np.diag([-1.0, 1.0, -1.0])


# ----------------------------------------------------------------
# Geometry from the .expt
# ----------------------------------------------------------------

def _rotate_onto(a, b):
    """Rotation matrix turning unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(a @ b)
    if s < 1e-15:
        if c > 0:
            return np.eye(3)
        raise ValueError("beam is antiparallel to z")
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def read_geometry(expt_file, experiment=0):
    """
    Collect what NXmx needs from one experiment of a .expt file, in
    the dxtbx frame.  Returns a dict.
    """
    with open(expt_file) as fh:
        data = json.load(fh)
    exp = data["experiment"][experiment]
    beam = data["beam"][exp["beam"]]
    panels = data["detector"][exp["detector"]]["panels"]
    if len(panels) != 1:
        raise NotImplementedError("Only single-panel detectors are supported.")
    panel = panels[0]
    gonio = data["goniometer"][exp["goniometer"]]
    scan = data["scan"][exp["scan"]]

    if "axes" in gonio:
        axes = [np.array(a, float) for a in gonio["axes"]]
        angles = [float(a) for a in gonio["angles"]]
        names = list(gonio.get("names") or
                     [f"axis{i}" for i in range(len(axes))])
        scan_axis = int(gonio["scan_axis"])
    else:
        S = np.array(gonio["setting_rotation"]).reshape(3, 3)
        F = np.array(gonio["fixed_rotation"]).reshape(3, 3)
        if not (np.allclose(S, np.eye(3)) and np.allclose(F, np.eye(3))):
            raise NotImplementedError(
                "Single-axis goniometer with non-identity setting or "
                "fixed rotation is not supported.")
        axes = [np.array(gonio["rotation_axis"], float)]
        angles = [0.0]
        names = ["omega"]
        scan_axis = 0

    props = scan["properties"]
    return dict(
        wavelength=float(beam["wavelength"]),
        direction=np.array(beam["direction"], float),   # sample -> source
        fast=np.array(panel["fast_axis"], float),
        slow=np.array(panel["slow_axis"], float),
        origin=np.array(panel["origin"], float),
        pixel_size=tuple(float(x) for x in panel["pixel_size"]),
        image_size=tuple(int(x) for x in panel["image_size"]),   # (nx, ny)
        trusted_range=tuple(float(x) for x in panel["trusted_range"]),
        material=panel.get("material") or "Si",
        thickness=float(panel.get("thickness") or 0.45),
        axes=axes, angles=angles, names=names, scan_axis=scan_axis,
        osc_start=np.array(props["oscillation"], float),
        osc_width=float(props["oscillation_width"][0])
        if "oscillation_width" in props
        else float(props["oscillation"][1] - props["oscillation"][0]),
        exposure=float(props["exposure_time"][0])
        if props.get("exposure_time") else None,
    )


def beam_alignment(geom):
    """Rotation that puts the beam direction exactly on +z (dxtbx)."""
    return _rotate_onto(geom["direction"], [0.0, 0.0, 1.0])


# ----------------------------------------------------------------
# Writing
# ----------------------------------------------------------------

def _vector_dataset(group, name, value, units, vector, depends_on,
                    transformation_type, offset=None):
    ds = group.create_dataset(name, data=np.atleast_1d(
        np.asarray(value, dtype="f8")))
    ds.attrs["units"] = units
    ds.attrs["transformation_type"] = transformation_type
    ds.attrs["vector"] = np.asarray(vector, dtype="f8")
    ds.attrs["depends_on"] = depends_on
    if offset is not None:
        ds.attrs["offset"] = np.asarray(offset, dtype="f8")
        ds.attrs["offset_units"] = "mm"
    return ds


def write_nxmx(path, images, geom, image_indices, probe="electron",
               compression="gzip"):
    """
    Write an NXmx file.

    images        : array (n, ny, nx), integer counts
    geom          : dict from read_geometry
    image_indices : the 0-based scan image index of each frame
                    (contiguous), used to pick the rotation angles
    """
    images = np.asarray(images)
    n, ny, nx = images.shape
    if (nx, ny) != geom["image_size"]:
        raise ValueError(f"image size {(nx, ny)} does not match the "
                         f"detector {geom['image_size']}")
    image_indices = list(image_indices)
    if image_indices != list(range(image_indices[0],
                                   image_indices[0] + n)):
        raise ValueError("frames must be a contiguous block of the scan")

    Q = beam_alignment(geom)
    to_nx = IMGCIF_TO_MCSTAS @ Q             # dxtbx lab -> NeXus lab
    fast = to_nx @ geom["fast"]
    slow = to_nx @ geom["slow"]
    origin = to_nx @ geom["origin"]

    now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    with h5py.File(path, "w") as f:
        f.attrs["NX_class"] = "NXroot"
        f.attrs["creator"] = "Medley"
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry["definition"] = np.bytes_("NXmx")
        entry["start_time"] = now
        entry["end_time_estimated"] = now

        source = entry.create_group("source")
        source.attrs["NX_class"] = "NXsource"
        source["name"] = "Medley simulation"
        source["probe"] = probe

        instrument = entry.create_group("instrument")
        instrument.attrs["NX_class"] = "NXinstrument"
        instrument["name"] = "Medley simulation"

        beam = instrument.create_group("beam")
        beam.attrs["NX_class"] = "NXbeam"
        wl = beam.create_dataset("incident_wavelength",
                                 data=geom["wavelength"])
        wl.attrs["units"] = "angstrom"

        # -- detector ---------------------------------------------
        det = instrument.create_group("detector")
        det.attrs["NX_class"] = "NXdetector"
        det["description"] = "Simulated by Medley"
        det["sensor_material"] = geom["material"]
        th = det.create_dataset("sensor_thickness",
                                data=geom["thickness"] * 1000.0)
        th.attrs["units"] = "microns"
        det.create_dataset("underload_value",
                           data=int(geom["trusted_range"][0]))
        det.create_dataset("saturation_value",
                           data=int(geom["trusted_range"][1]))
        det.create_dataset("bit_depth_readout", data=32)
        if geom["exposure"] is not None:
            ft = det.create_dataset("frame_time", data=geom["exposure"])
            ft.attrs["units"] = "s"

        dtr = det.create_group("transformations")
        dtr.attrs["NX_class"] = "NXtransformations"
        # A zero translation as the root of the detector chain; the
        # panel position is carried by the module offset below.
        _vector_dataset(dtr, "det_z", 0.0, "mm", [0.0, 0.0, 1.0], ".",
                        "translation")
        det["depends_on"] = "/entry/instrument/detector/transformations/det_z"

        module = det.create_group("module")
        module.attrs["NX_class"] = "NXdetector_module"
        module.create_dataset("data_origin", data=np.array([0, 0], "i4"))
        module.create_dataset("data_size", data=np.array([ny, nx], "i4"))
        root = "/entry/instrument/detector/transformations/det_z"
        _vector_dataset(module, "fast_pixel_direction",
                        geom["pixel_size"][0], "mm", fast, root,
                        "translation", offset=origin)
        _vector_dataset(module, "slow_pixel_direction",
                        geom["pixel_size"][1], "mm", slow, root,
                        "translation", offset=origin)

        # -- sample / goniometer ----------------------------------
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        str_ = sample.create_group("transformations")
        str_.attrs["NX_class"] = "NXtransformations"

        start = geom["osc_start"][image_indices]
        width = geom["osc_width"]
        names = geom["names"]
        for i, (axis, ang, name) in enumerate(zip(
                geom["axes"], geom["angles"], names)):
            vec = to_nx @ axis
            dep = ("." if i == len(names) - 1
                   else f"/entry/sample/transformations/{names[i + 1]}")
            if i == geom["scan_axis"]:
                ds = _vector_dataset(str_, name, start, "deg", vec, dep,
                                     "rotation")
                inc = str_.create_dataset(f"{name}_increment_set",
                                          data=np.full(n, width))
                inc.attrs["units"] = "deg"
                end = str_.create_dataset(f"{name}_end", data=start + width)
                end.attrs["units"] = "deg"
            else:
                _vector_dataset(str_, name, [ang], "deg", vec, dep,
                                "rotation")
        sample["depends_on"] = f"/entry/sample/transformations/{names[0]}"

        # -- data -------------------------------------------------
        nxdata = entry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "data"
        nxdata.create_dataset("data", data=images.astype(np.int32),
                              chunks=(1, ny, nx), compression=compression)
    return Q
