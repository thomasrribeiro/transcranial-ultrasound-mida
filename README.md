# Transcranial ultrasound through MIDA

A small Python example that maps the segmented **MIDA v1 human head** to sound
speed, density, and attenuation, then runs a **3D k-Wave** pressure simulation.
A rectangular aperture emits a simultaneous tone burst from the high side of
MIDA axis 2. The output is the acoustic volume and a central pressure plane over
time. This is a fresh, minimal implementation inspired by
[alephneuro/transcranial-ultrasound-mida](https://github.com/alephneuro/transcranial-ultrasound-mida).

## Install and run

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/thomasrribeiro/transcranial-ultrasound-mida.git
cd transcranial-ultrasound-mida
uv sync --locked
```

Obtain the **MIDA v1 voxel model** from the
[IT'IS MIDA download page](https://itis.swiss/virtual-population/regional-human-models/mida-model).
Use `MIDA_v1_voxels/MIDA_v1.mat`, containing the integer `tissuedistrib` array at
0.5 mm spacing. The anatomical dataset is not bundled; obtain it under its own
license. The acoustic label table is included as `acoustic_properties.csv`.

Start with this small temporal-region crop (3D, approximately 20 × 20 × 30 mm):

```bash
uv run --no-sync python simulate.py /path/to/MIDA_v1.mat \
  --crop 230:270 220:260 285:345 \
  --aperture-mm 10 10 --duration-us 30 \
  --output outputs/temporal
```

Defaults: 500 kHz, 3 cycles, 100 kPa source amplitude, 0.5 mm grid, CFL 0.2,
1.5 mm water standoff, 12-cell external PML, NumPy CPU solver. This small crop
demonstrates entry through the skull; expand the crop and duration to study
deeper propagation. Crop indices are zero-based, stop-exclusive, in the original
MATLAB array axis order. Omitting `--crop` uses the entire model and can require
substantial memory and a finer grid for air cavities.

Use `--dry-run --output outputs/check` to save the acoustic maps and inspect
settings before solving. Every run needs a new output directory. The CLI prints
the pressure-recording allocation; this is **not** the total solver memory.

The stock `k-wave-python==0.6.2` package is pinned in `uv.lock`; no custom fork is
needed. It may download its solver binaries on first import, even for the Python
backend. For NVIDIA GPU execution with the Python backend, install the CuPy
package matching your CUDA installation (for CUDA 12, `uv pip install cupy-cuda12x`)
and add `--device gpu`. Keep using `uv run --no-sync` to retain that extra.
`--backend cpp` selects k-Wave's compiled backend instead; its binary/runtime
requirements depend on the platform. CPU Python execution is the tested baseline.

## Acoustic map and assumptions

- `acoustic_properties.csv` explicitly maps labels 0–116 to sound speed (m/s),
  density (kg/m³), attenuation (dB/cm/MHz^b), and exponent `b`. Values were resolved
  from the original project's IT'IS V5.0 property export. Each row names its
  source tissue; the eye composite fallback and water coupling values are marked
  as approximations. See the [IT'IS tissue database](https://itis.swiss/virtual-population/tissue-properties/database/).
  Supply `--properties your.csv` to change assumptions. Unknown labels fail.
- MIDA background (labels 0 and 50) is coupling water. Internal air is preserved.
  `--air water` explicitly replaces air cavities with water and records that
  approximation. It changes the physics and should only be used intentionally.
- Label resampling uses nearest neighbours on the requested physical grid;
  tissue interfaces are not blended. Coordinates are relative to the centre of
  native voxel `(0, 0, 0)`; anatomical orientation is not inferred or transposed.
- k-Wave uses a single absorption exponent, default `y = 1.5`. Each coefficient
  is converted using `alpha_eff = alpha_table * f_MHz**(b-y)` so attenuation
  matches the input tissue law at the carrier. Broadband attenuation is therefore
  approximate. `--alpha-power` changes the common exponent (excluding 1).
- This is a linear **fluid-acoustic** model: it does not simulate skull shear
  waves, heating, microbubbles, or elastic mode conversion. The additive pressure
  sheet radiates in both directions; the outer PML absorbs outward propagation.
- The grid check requires at least 3 points per wavelength at the carrier in
  the slowest material. This is a coarse screening check, not a convergence
  study; broadband pulses and sharp interfaces need resolution/PML convergence
  checks for quantitative work. Keeping internal air typically needs a finer
  grid (e.g. `--dx-mm 0.2` at 500 kHz).

## Outputs

Each run writes:

- `medium.npz`: labels, 3D sound speed/density/attenuation, source mask, grid
  spacing and origin. The high-axis-2 water padding is included; the PML is not.
- `pressure.npz`: `pressure_pa[time, axis1, axis2]` on the central axis-0 plane,
  `times_s`, and physical coordinates in metres. This is a slice of a 3D solve,
  not a 2D approximation. No per-frame normalization is applied.
- `settings.json`: run arguments, input hashes, time step, grid and source
  location, and resolved tissue properties.

```python
import numpy as np

with np.load("outputs/temporal/pressure.npz") as result:
    pressure = result["pressure_pa"]   # Pa, shape (time, axis1, axis2)
    peak_absolute_pressure = np.max(np.abs(pressure), axis=0)
```

Run checks with `uv run --no-sync python -m unittest -v`. They cover map units,
attenuation conversion, label/coordinate preservation, source clearance, and an
actual homogeneous-water k-Wave solve checking propagation speed and sensor order.
