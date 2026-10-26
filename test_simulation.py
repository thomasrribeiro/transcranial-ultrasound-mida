"""Mapping, coordinates, source placement, and a real k-Wave water propagation check."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat
from scipy.signal import correlate, correlation_lags

from mida import acoustic_maps, add_source_space, load_labels, read_properties


class MidaTests(unittest.TestCase):
    def test_attenuation_matches_original_at_carrier(self):
        table = read_properties()
        labels = np.array([[[10, 40, 52, 50]]], dtype=np.uint16)
        (_, _, alpha), _ = acoustic_maps(labels, table, 500e3)
        expected = [table[i]["alpha_db_cm_mhz_b"] * 0.5 ** table[i]["b"] for i in labels.ravel()]
        np.testing.assert_allclose(alpha.ravel() * 0.5**1.5, expected, rtol=1e-6)
        self.assertLess(table[52]["sound_speed_m_s"], table[40]["sound_speed_m_s"])

    def test_missing_labels_fail_and_air_is_explicit(self):
        table = read_properties()
        with self.assertRaisesRegex(ValueError, "Missing acoustic"):
            acoustic_maps(np.array([[[999]]]), table, 500e3)
        labels = np.array([[[26]]])
        maps, _ = acoustic_maps(labels, table, 500e3)
        self.assertEqual(maps[0].item(), 343)
        maps, used = acoustic_maps(labels, table, 500e3, air="water")
        self.assertAlmostEqual(maps[0].item(), table[50]["sound_speed_m_s"], places=3)
        self.assertTrue(used[0]["air_replaced_with_water"])

    def test_resampling_preserves_labels_and_exact_coordinates(self):
        v = np.full((12, 14, 16), 50, dtype=np.uint16)
        v[6:, :, :] = 40
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.mat"
            savemat(path, {"tissuedistrib": v})
            sampled, origin = load_labels(path, [(2, 11), (3, 13), (4, 15)], 0.3e-3)
            np.testing.assert_allclose(origin, [0.001, 0.0015, 0.002])
            self.assertEqual(set(np.unique(sampled)), {40, 50})
            expected_native = 2 + np.rint(np.arange(sampled.shape[0]) * 0.3 / 0.5)
            np.testing.assert_array_equal(sampled[:, 0, 0], np.where(expected_native >= 6, 40, 50))
            with self.assertRaisesRegex(ValueError, "outside"):
                load_labels(path, [(0, 99), (0, 14), (0, 16)])

    def test_source_clears_highest_tissue_and_padding(self):
        v = np.full((20, 20, 20), 50, dtype=np.uint16)
        v[5:15, 5:15, :18] = 40
        padded, mask = add_source_space(v, 0.5e-3, [4e-3, 4e-3], 1.5e-3)
        self.assertTrue((padded[mask] == 50).all())
        self.assertEqual(mask.sum(), 64)
        self.assertEqual(np.nonzero(mask)[2][0], 20)
        self.assertGreater(padded.shape[2], 20)


class SolverTests(unittest.TestCase):
    def test_water_speed_and_sensor_axis_order(self):
        from kwave.kgrid import kWaveGrid
        from kwave.kmedium import kWaveMedium
        from kwave.ksensor import kSensor
        from kwave.ksource import kSource
        from kwave.kspaceFirstOrder import kspaceFirstOrder
        from kwave.utils.signals import tone_burst

        grid = kWaveGrid([20, 20, 48], [0.5e-3] * 3)
        grid.makeTime(1500, 0.2, 16e-6)
        source = kSource()
        source.p_mask = np.zeros((20, 20, 48), dtype=bool)
        source.p_mask[:, :, 5] = True
        source.p = tone_burst(1 / grid.dt, 500e3, 3) * 1e3
        mask = np.zeros((20, 20, 48), dtype=bool)
        mask[10, 10, [13, 21]] = True
        result = kspaceFirstOrder(grid, kWaveMedium(sound_speed=1500, density=1000), source,
                                 kSensor(mask=mask, record=["p"]), pml_size=8,
                                 backend="python", dtype=np.float32, quiet=True)
        p = result["p"]
        self.assertEqual(p.shape, (2, int(grid.Nt)))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreater(np.abs(p).max(), 10)
        delay = correlation_lags(p.shape[1], p.shape[1])[np.argmax(correlate(p[1], p[0]))]
        speed = (21 - 13) * 0.5e-3 / (delay * grid.dt)
        self.assertAlmostEqual(speed, 1500, delta=75)


if __name__ == "__main__":
    unittest.main()
