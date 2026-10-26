"""MIDA v1 labels to acoustic maps; all lengths passed to functions are metres."""
from pathlib import Path
import csv

import numpy as np
from scipy.io import loadmat

PROPERTIES = Path(__file__).with_name("acoustic_properties.csv")
NATIVE_DX = 0.5e-3
BACKGROUND = (0, 50)


def read_properties(path=PROPERTIES):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    table = {}
    for row in rows:
        label = int(row["label"])
        if label in table or label < 0:
            raise ValueError(f"Duplicate or negative label: {label}")
        for key in ("sound_speed_m_s", "density_kg_m3", "alpha_db_cm_mhz_b", "b"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]) or row[key] < 0:
                raise ValueError(f"Invalid {key} for label {label}")
        if min(row["sound_speed_m_s"], row["density_kg_m3"]) <= 0:
            raise ValueError(f"Sound speed and density must be positive: {label}")
        table[label] = row
    if not all(i in table for i in BACKGROUND):
        raise ValueError("Property table must include water/background labels 0 and 50")
    return table


def load_labels(path, crop=None, dx=NATIVE_DX):
    """Nearest-neighbour labels on an exact dx lattice, preserving MATLAB axis order.

    Voxel zero is the centre of the first native voxel. Crop stops are exclusive.
    Native and output coordinates coincide at the crop's first voxel centre.
    """
    volume = loadmat(path, variable_names=["tissuedistrib"])["tissuedistrib"]
    if volume.ndim != 3 or not np.issubdtype(volume.dtype, np.integer):
        raise ValueError("tissuedistrib must be a 3D integer label array")
    if np.any(volume < 0) or volume.max() > np.iinfo(np.uint16).max:
        raise ValueError("Labels must fit unsigned 16-bit integers")
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("dx must be finite and positive")
    bounds = crop or [(0, n) for n in volume.shape]
    if len(bounds) != 3 or any(not 0 <= a < b <= n for (a, b), n in zip(bounds, volume.shape)):
        raise ValueError(f"Crop is outside input shape {volume.shape}")
    labels = volume[tuple(slice(a, b) for a, b in bounds)]
    # Do not use zoom(): its endpoint alignment changes the effective spacing.
    coords = [np.arange(int(np.floor((n - 1) * NATIVE_DX / dx + 1e-9)) + 1) * dx / NATIVE_DX
              for n in labels.shape]
    indices = [np.rint(x).astype(int) for x in coords]
    sampled = labels[np.ix_(*indices)].astype(np.uint16)
    if min(sampled.shape) < 4:
        raise ValueError("Crop/resampling leaves fewer than four cells on an axis")
    return sampled, np.array([a for a, _ in bounds]) * NATIVE_DX


def acoustic_maps(labels, table, frequency_hz, alpha_power=1.5, air="preserve"):
    """Match each tissue's attenuation at the carrier using one k-Wave exponent.

    alpha_eff * f_MHz**alpha_power = alpha_table * f_MHz**b.
    No tissue interpolation, invented scatterers, or silent missing-label fallback.
    """
    if frequency_hz <= 0 or not 0 < alpha_power < 3 or alpha_power == 1:
        raise ValueError("Use positive frequency and 0 < alpha_power < 3, excluding 1")
    if air not in ("preserve", "water"):
        raise ValueError("air must be preserve or water")
    ids = np.unique(labels)
    missing = set(map(int, ids)) - table.keys()
    if missing:
        raise ValueError(f"Missing acoustic properties for labels {sorted(missing)}")
    lut = np.zeros((3, int(ids.max()) + 1), dtype=np.float32)
    used = []
    for idx in ids:
        row = table[int(idx)]
        replaced = air == "water" and row["tissue"].startswith("Air Internal")
        props = table[50] if replaced else row
        alpha = props["alpha_db_cm_mhz_b"] * (frequency_hz / 1e6) ** (props["b"] - alpha_power)
        lut[:, idx] = props["sound_speed_m_s"], props["density_kg_m3"], alpha
        used.append({**row, "air_replaced_with_water": replaced,
                     "effective_alpha_db_cm_mhz_y": alpha})
    return tuple(lut[k][labels] for k in range(3)), used


def add_source_space(labels, dx, aperture_m, standoff_m):
    """Put a flat rectangular source above all tissue under its footprint (+axis 2)."""
    width = np.maximum(1, np.rint(np.asarray(aperture_m) / dx).astype(int))
    if np.any(width > np.array(labels.shape[:2]) - 2):
        raise ValueError("Aperture must leave at least one cell at either lateral edge")
    start = (np.array(labels.shape[:2]) - width) // 2
    footprint = tuple(slice(int(a), int(a + n)) for a, n in zip(start, width))
    tissue = ~np.isin(labels[footprint], BACKGROUND)
    occupied = np.flatnonzero(tissue.any(axis=(0, 1)))
    if not len(occupied):
        raise ValueError("No head tissue beneath the source aperture")
    source_z = int(occupied[-1]) + max(1, int(np.ceil(standoff_m / dx)))
    nz = max(labels.shape[2], source_z + 1 + int(np.ceil(3e-3 / dx)))
    padded = np.pad(labels, ((0, 0), (0, 0), (0, nz - labels.shape[2])), constant_values=50)
    mask = np.zeros(padded.shape, dtype=bool)
    mask[footprint + (source_z,)] = True
    if not np.isin(padded[mask], BACKGROUND).all():
        raise ValueError("Source overlaps tissue")
    return padded, mask
