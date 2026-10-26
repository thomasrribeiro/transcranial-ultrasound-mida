"""Run a 3D acoustic simulation and save a central pressure movie as NumPy arrays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mida import PROPERTIES, acoustic_maps, add_source_space, load_labels, read_properties


def crop_range(value):
    try:
        a, b = map(int, value.split(":"))
        if 0 <= a < b:
            return a, b
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Use a nonnegative start:stop range (stop excluded)")


def positive(value):
    number = float(value)
    if not np.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive")
    return number


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path, help="MIDA v1 .mat containing tissuedistrib")
    p.add_argument("--properties", type=Path, default=PROPERTIES)
    p.add_argument("--crop", nargs=3, type=crop_range, metavar=("AXIS0", "AXIS1", "AXIS2"))
    p.add_argument("--dx-mm", type=positive, default=0.5)
    p.add_argument("--frequency-khz", type=positive, default=500)
    p.add_argument("--cycles", type=positive, default=3)
    p.add_argument("--pressure-pa", type=positive, default=1e5)
    p.add_argument("--aperture-mm", nargs=2, type=positive, default=[10, 10])
    p.add_argument("--standoff-mm", type=positive, default=1.5)
    p.add_argument("--duration-us", type=positive, default=60)
    p.add_argument("--air", choices=["preserve", "water"], default="preserve")
    p.add_argument("--alpha-power", type=positive, default=1.5)
    p.add_argument("--backend", choices=["python", "cpp"], default="python")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--output", type=Path, default=Path("outputs/head"))
    p.add_argument("--dry-run", action="store_true", help="Save maps and run settings without solving")
    return p


def sha256(path):
    with open(path, "rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(args):
    dx = args.dx_mm * 1e-3
    frequency = args.frequency_khz * 1e3
    labels, origin = load_labels(args.model, args.crop, dx)
    labels, source_mask = add_source_space(labels, dx, np.array(args.aperture_mm) * 1e-3,
                                            args.standoff_mm * 1e-3)
    if min(labels.shape) <= 24:
        raise ValueError("Use at least 25 grid cells per axis (k-Wave's 12-cell PML validation)")
    (c, rho, alpha), tissues = acoustic_maps(labels, read_properties(args.properties),
                                            frequency, args.alpha_power, args.air)
    ppw = float(c.min()) / (frequency * dx)
    if ppw < 3:
        raise ValueError(f"Only {ppw:.2f} points/wavelength in the slowest tissue; reduce --dx-mm "
                         f"to <= {float(c.min()) / frequency / 3 * 1e3:.3g}. "
                         "--air water is an explicit approximation, if desired.")
    # Import only after input validation; k-Wave can download binaries on first import.
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder import kspaceFirstOrder
    from kwave.utils.signals import tone_burst

    grid = kWaveGrid(labels.shape, [dx] * 3)
    grid.makeTime(float(c.max()), 0.2, args.duration_us * 1e-6)
    signal = np.asarray(tone_burst(1 / grid.dt, frequency, args.cycles), dtype=np.float32)
    signal *= args.pressure_pa
    if signal.shape[-1] >= grid.Nt:
        raise ValueError("Duration must be longer than the source pulse")
    source = kSource()
    source.p_mask = source_mask
    source.p = signal
    source.p_mode = "additive"
    sensor_mask = np.zeros(labels.shape, dtype=bool)
    plane_index = labels.shape[0] // 2
    sensor_mask[plane_index, :, :] = True
    sensor = kSensor(mask=sensor_mask, record=["p"])
    medium = kWaveMedium(sound_speed=c, density=rho, alpha_coeff=alpha, alpha_power=args.alpha_power)
    settings = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_sha256": sha256(args.model), "properties_sha256": sha256(args.properties),
        "kwave_version": "0.6.2", "shape": list(labels.shape), "origin_m": origin.tolist(),
        "dx_m": dx, "dt_s": float(grid.dt), "nt": int(grid.Nt), "cfl": 0.2,
        "pml_cells_outside": 12, "minimum_carrier_points_per_wavelength": ppw,
        "sensor_axis0_index": plane_index, "source_axis2_index": int(np.nonzero(source_mask)[2][0]),
        "pressure_recording_gib": int(sensor_mask.sum()) * int(grid.Nt) * 4 / 2**30,
        "tissues": tissues,
    }
    # Refuse accidental overwrite of previous results.
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / f).exists() for f in ("medium.npz", "pressure.npz", "settings.json")):
        raise ValueError("Output already contains a run; choose a new --output directory")
    np.savez_compressed(args.output / "medium.npz", labels=labels, sound_speed_m_s=c,
                        density_kg_m3=rho, alpha_db_cm_mhz_y=alpha, alpha_power=args.alpha_power,
                        dx_m=dx, origin_m=origin, source_mask=source_mask)
    (args.output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(f"Grid {labels.shape}, {grid.Nt} steps, {ppw:.2f} points/wavelength; "
          f"pressure recording alone: {settings['pressure_recording_gib']:.2f} GiB", flush=True)
    if args.dry_run:
        print(f"Saved acoustic maps and settings to {args.output}")
        return
    result = kspaceFirstOrder(grid, medium, source, sensor, pml_inside=False, pml_size=12,
                             backend=args.backend, device=args.device,
                             dtype=np.float32 if args.backend == "python" else None)
    # Unified k-Wave API returns (sensor, time), sensors in C-flattened mask order.
    pressure = np.asarray(result["p"])
    expected = (int(sensor_mask.sum()), int(grid.Nt))
    if pressure.shape != expected or not np.isfinite(pressure).all():
        raise RuntimeError(f"Invalid solver output: shape {pressure.shape}, expected {expected}")
    plane = pressure.T.reshape((int(grid.Nt), *labels.shape[1:]))
    np.savez_compressed(args.output / "pressure.npz", pressure_pa=plane.astype(np.float32),
                        times_s=np.asarray(grid.t_array).ravel(), dx_m=dx,
                        axis0_m=origin[0] + plane_index * dx,
                        axis1_m=origin[1] + np.arange(labels.shape[1]) * dx,
                        axis2_m=origin[2] + np.arange(labels.shape[2]) * dx)
    print(f"Saved {args.output / 'pressure.npz'}; peak |p| = {np.abs(pressure).max():.3g} Pa")


if __name__ == "__main__":
    cli = parser()
    try:
        run(cli.parse_args())
    except (ValueError, KeyError, FileNotFoundError) as error:
        cli.exit(2, f"Error: {error}\n")
