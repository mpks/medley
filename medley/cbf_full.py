"""
cbf_full.py — write a full imgCIF CBF that encodes the detector
and beam geometry explicitly.

Unlike the miniCBF (PILATUS/Eiger) header, which forces dxtbx to
recognise a specific detector model and reconstruct its module
layout, a full CBF stores the panel basis (fast axis, slow axis,
origin), pixel sizes and array dimensions directly in imgCIF
categories.  dxtbx reads this with FormatCBFFull for a panel of
any size, so it works for detectors that are not a standard
Eiger/Pilatus (e.g. 775 x 385 panels), avoiding the Eiger
module-mask lookup that fails for non-standard sizes.

The image is stored with CBF_BYTE_OFFSET compression.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np


# ----------------------------------------------------------------
# CBF_BYTE_OFFSET compression
# ----------------------------------------------------------------

def byte_offset_compress(data):
    """
    Compress a 2D int array with the CBF_BYTE_OFFSET scheme.

    Consecutive pixel differences are stored as 1 byte when they
    fit in [-127, 127]; otherwise an escape 0x80 is written and
    the difference is stored as int16, with further escapes to
    int32 and int64 for larger jumps.
    """
    flat = data.astype(np.int64).ravel()
    out = bytearray()
    prev = 0
    for v in flat:
        delta = int(v) - prev
        prev = int(v)
        if -127 <= delta <= 127:
            out += struct.pack("<b", delta)
            continue
        out += b"\x80"
        if -32767 <= delta <= 32767:
            out += struct.pack("<h", delta)
            continue
        out += struct.pack("<h", -32768)
        if -2147483647 <= delta <= 2147483647:
            out += struct.pack("<i", delta)
            continue
        out += struct.pack("<i", -2147483648)
        out += struct.pack("<q", delta)
    return bytes(out)


# ----------------------------------------------------------------
# imgCIF header
# ----------------------------------------------------------------

def _axis_block(detector, beam):
    """
    Build the _axis category describing the goniometer, the
    detector translation/rotation and the two in-plane detector
    axes.  The detector fast/slow axes and origin are taken from
    the Detector; the source axis from the beam direction.

    Geometry follows the CBF/imgCIF convention: axes are given
    in the laboratory frame, the detector origin is the position
    of pixel (0, 0).
    """
    fast = np.asarray(detector.fast_axis, dtype=float)
    slow = np.asarray(detector.slow_axis, dtype=float)
    origin = np.asarray(detector.origin, dtype=float)
    src = np.asarray(beam.direction, dtype=float)
    src = src / np.linalg.norm(src)

    lines = [
        "loop_",
        "_axis.id",
        "_axis.type",
        "_axis.equipment",
        "_axis.depends_on",
        "_axis.vector[1] _axis.vector[2] _axis.vector[3]",
        "_axis.offset[1] _axis.offset[2] _axis.offset[3]",
        # source axis (beam propagation direction)
        (f"SOURCE      general source . "
         f"{src[0]:.8f} {src[1]:.8f} {src[2]:.8f}  0 0 0"),
        # gravity (unused but conventional)
        "GRAVITY     general gravity . 0 -1 0  0 0 0",
        # detector translation to the panel origin
        (f"DETECTOR_Z  translation detector . 0 0 1  "
         f"{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}"),
        # fast and slow in-plane axes, depending on the detector
        (f"ELEMENT_X   translation detector DETECTOR_Z  "
         f"{fast[0]:.8f} {fast[1]:.8f} {fast[2]:.8f}  0 0 0"),
        (f"ELEMENT_Y   translation detector ELEMENT_X  "
         f"{slow[0]:.8f} {slow[1]:.8f} {slow[2]:.8f}  0 0 0"),
    ]
    return "\n".join(lines)


def build_full_header(detector, beam, start_angle_deg,
                      angle_increment_deg):
    """
    Build the imgCIF header text (everything before the binary
    section) encoding the detector panel geometry, pixel sizes,
    array dimensions and the radiation wavelength.
    """
    npx = detector.npx
    npy = detector.npy
    px_mm = detector.pixel_size_mm
    px_m = px_mm * 1e-3
    wl = beam.wavelength_A

    parts = []
    parts.append("_diffrn.id DS1")
    parts.append("_diffrn.crystal_id CRYSTAL1")
    parts.append("")
    parts.append("_diffrn_source.diffrn_id DS1")
    parts.append("_diffrn_source.source electron")
    parts.append("")
    parts.append("_diffrn_radiation.diffrn_id DS1")
    parts.append(
        "_diffrn_radiation.wavelength_id WAVELENGTH1"
    )
    parts.append("_diffrn_radiation.probe electron")
    parts.append("")
    parts.append("loop_")
    parts.append("_diffrn_radiation_wavelength.id")
    parts.append("_diffrn_radiation_wavelength.wavelength")
    parts.append("_diffrn_radiation_wavelength.wt")
    parts.append(f"WAVELENGTH1 {wl:.6f} 1.0")
    parts.append("")
    parts.append("_diffrn_detector.diffrn_id DS1")
    parts.append("_diffrn_detector.id DETECTOR1")
    parts.append("_diffrn_detector.number_of_axes 3")
    parts.append("")
    # Scan / oscillation info (single frame).
    parts.append("_diffrn_scan.id SCAN1")
    parts.append("_diffrn_scan.frame_id FRAME1")
    parts.append("_diffrn_scan.frames 1")
    parts.append("")
    parts.append("_diffrn_scan_frame.frame_id FRAME1")
    parts.append("_diffrn_scan_frame.scan_id SCAN1")
    parts.append("_diffrn_scan_frame.frame_number 1")
    parts.append(
        f"_diffrn_scan_frame.integration_time 1.0"
    )
    parts.append("")
    parts.append(_axis_block(detector, beam))
    parts.append("")
    # Array structure: two dimensions, fast then slow.
    parts.append("loop_")
    parts.append("_array_structure_list.array_id")
    parts.append("_array_structure_list.index")
    parts.append("_array_structure_list.dimension")
    parts.append("_array_structure_list.precedence")
    parts.append("_array_structure_list.direction")
    parts.append("_array_structure_list.axis_set_id")
    parts.append(f"IMAGE 1 {npx} 1 increasing ELEMENT_X")
    parts.append(f"IMAGE 2 {npy} 2 increasing ELEMENT_Y")
    parts.append("")
    parts.append("loop_")
    parts.append("_array_structure_list_axis.axis_set_id")
    parts.append("_array_structure_list_axis.axis_id")
    parts.append(
        "_array_structure_list_axis.displacement"
    )
    parts.append(
        "_array_structure_list_axis.displacement_increment"
    )
    parts.append(
        f"ELEMENT_X ELEMENT_X {0.5 * px_m:.9f} {px_m:.9f}"
    )
    parts.append(
        f"ELEMENT_Y ELEMENT_Y {0.5 * px_m:.9f} {px_m:.9f}"
    )
    parts.append("")
    parts.append("_diffrn_frame_data.id FRAME1")
    parts.append("_diffrn_frame_data.detector_element_id ELEMENT1")
    parts.append("_diffrn_frame_data.array_id IMAGE")
    parts.append("_diffrn_frame_data.binary_id 1")
    parts.append("")
    return "\n".join(parts)


# ----------------------------------------------------------------
# Full writer
# ----------------------------------------------------------------

def write_full_cbf(image_int32, detector, beam,
                   start_angle_deg, angle_increment_deg,
                   output_path):
    """
    Write a full imgCIF CBF with explicit geometry and
    byte-offset compressed data.
    """
    npy, npx = image_int32.shape
    header = build_full_header(
        detector, beam, start_angle_deg, angle_increment_deg
    )
    compressed = byte_offset_compress(image_int32)
    digest = hashlib.md5(compressed).digest()
    import base64
    md5_b64 = base64.b64encode(digest).decode("ascii")

    text_head = (
        "###CBF: VERSION 1.0\n"
        "# Written by dynamic.simulation (full imgCIF)\n\n"
        "data_frame1\n\n"
        + header
        + "\n_array_data.data\n"
        ";\n"
        "--CIF-BINARY-FORMAT-SECTION--\n"
        "Content-Type: application/octet-stream;\n"
        '     conversions="x-CBF_BYTE_OFFSET"\n'
        "Content-Transfer-Encoding: BINARY\n"
        f"X-Binary-Size: {len(compressed)}\n"
        "X-Binary-ID: 1\n"
        'X-Binary-Element-Type: "signed 32-bit integer"\n'
        "X-Binary-Element-Byte-Order: LITTLE_ENDIAN\n"
        f"X-Binary-Number-of-Elements: {npx * npy}\n"
        f"X-Binary-Size-Fastest-Dimension: {npx}\n"
        f"X-Binary-Size-Second-Dimension: {npy}\n"
        f"Content-MD5: {md5_b64}\n"
        "\n"
    )

    with open(output_path, "wb") as f:
        f.write(text_head.encode("ascii"))
        # CBF binary identifier: 0x0C 0x1A 0x04 0xD5
        f.write(b"\x0c\x1a\x04\xd5")
        f.write(compressed)
        f.write(
            b"\n--CIF-BINARY-FORMAT-SECTION----\n;\n"
        )
