"""Unit tests for run_sweep and per-run Abseil logging directory setup."""

import datetime
import pathlib
import tempfile
import unittest
import unittest.mock

from absl import logging
import run_sweep
from src import io_manager


class RunSweepAndLoggingTest(unittest.TestCase):
  """Tests for sweep execution and Abseil logging directory initialization."""

  def test_run_geometry_group_solves_and_writes_output(self) -> None:
    """Verifies _run_geometry_group solves a config and writes .out.textproto."""
    point = run_sweep.SweepPoint(
        s_kiln_mm=200.0,
        w_floor_pad_mm=1200.0,
        t_floor_pad_mm=0.0,
        w_door_open_mm=150.0,
        exhaust_fan_cfm=400.0,
        d_reflector_mm=800.0,
        t_reflector_calsil_mm=10.0,
        g_floor_reflector_mm=10.0,
        d_chimney_mm=0.0,
        g_chimney_mm=0.0,
        w_shield_mm=1200.0,
        d_hood_inlet_mm=800.0,
        bung_captured=True,
    )
    cfg = run_sweep.build_config_proto(point)
    # Use a coarser grid in the unit test for speed.
    cfg.grid.dx_wall_mm = 200.0
    cfg.grid.dz_wall_mm = 200.0
    cfg.grid.n_theta_kiln = 8
    cfg.grid.n_z_kiln = 3

    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = pathlib.Path(tmp_dir)
      config_path = tmp_path / f"{point.experiment_name}.textproto"
      output_dir = tmp_path / "results"
      output_dir.mkdir()
      run_sweep.write_config_textproto(cfg, config_path)

      attempted, succeeded, failed_runs = run_sweep._run_geometry_group(
          [str(config_path)], str(output_dir)
      )

      self.assertEqual(attempted, 1)
      self.assertEqual(failed_runs, [])
      self.assertEqual(succeeded, 1)
      expected_out = output_dir / f"{point.experiment_name}.out.textproto"
      self.assertTrue(expected_out.is_file())

  def test_configure_absl_logging_creates_timestamped_subdir(self) -> None:
    """Verifies configure_absl_logging creates yyyymmdd-hhmmss.s_<binary>."""
    fixed_dt = datetime.datetime(
        2026, 11, 3, 7, 53, 40, 600000, tzinfo=datetime.timezone.utc
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
      base_dir = pathlib.Path(tmp_dir)
      run_log_dir = io_manager.configure_absl_logging(
          binary_name="run_sweep",
          base_log_dir=base_dir,
          dt=fixed_dt,
      )
      logging.info("Test log entry inside unique subdirectory.")
      logging.get_absl_handler().flush()

      self.assertEqual(run_log_dir.name, "20261103-075340.6_run_sweep")
      self.assertTrue(run_log_dir.is_dir())
      log_files = list(run_log_dir.glob("*run_sweep*"))
      self.assertTrue(log_files)

  def test_reservoir_sample_points_for_quick_flag(self) -> None:
    """Verifies reservoir_sample_points selects 2*max_workers unique points."""
    all_points = run_sweep.enumerate_sweep_points()
    max_workers = 8
    target_k = 2 * max_workers
    sampled = run_sweep.reservoir_sample_points(all_points, target_k, seed=42)

    self.assertEqual(len(sampled), target_k)
    self.assertEqual(len(set(sampled)), target_k)
    for pt in sampled:
      self.assertIn(pt, all_points)
    self.assertIn("quick", run_sweep.flags.FLAGS)

  def test_run_geometry_group_logs_experiment_start_and_dt_room_on_failure(
      self,
  ) -> None:
    """Verifies start logging of experiment_name and dt_room_change in failure msg."""
    point = run_sweep.SweepPoint(
        s_kiln_mm=200.0,
        w_floor_pad_mm=1200.0,
        t_floor_pad_mm=0.0,
        w_door_open_mm=150.0,
        exhaust_fan_cfm=400.0,
        d_reflector_mm=800.0,
        t_reflector_calsil_mm=10.0,
        g_floor_reflector_mm=10.0,
        d_chimney_mm=0.0,
        g_chimney_mm=0.0,
        w_shield_mm=1200.0,
        d_hood_inlet_mm=800.0,
        bung_captured=True,
    )
    cfg = run_sweep.build_config_proto(point)
    cfg.grid.dx_wall_mm = 200.0
    cfg.grid.dz_wall_mm = 200.0
    cfg.grid.n_theta_kiln = 8
    cfg.grid.n_z_kiln = 3

    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = pathlib.Path(tmp_dir)
      config_path = tmp_path / f"{point.experiment_name}.textproto"
      output_dir = tmp_path / "results"
      output_dir.mkdir()
      run_sweep.write_config_textproto(cfg, config_path)

      with self.assertLogs(level="INFO") as cm:
        run_sweep._run_geometry_group([str(config_path)], str(output_dir))
      joined_logs = "\n".join(cm.output)
      self.assertIn(f"Starting experiment: {point.experiment_name}", joined_logs)

      # Verify SolverSolution exposes dt_room_change_k and failure summary formats it.
      fake_unconverged = unittest.mock.MagicMock()
      fake_unconverged.converged = False
      fake_unconverged.max_energy_residual_w_m2 = 8.2e-5
      fake_unconverged.dt_room_change_k = 0.280
      with (
          unittest.mock.patch.object(
              run_sweep.solver_lib,
              "solve_thermal_system",
              return_value=fake_unconverged,
          ),
          unittest.mock.patch.object(
              run_sweep.metrics_lib,
              "build_simulation_output",
              return_value=run_sweep.kiln_sim_pb2.SimulationOutput(),
          ),
      ):
        _, succeeded, failed = run_sweep._run_geometry_group(
            [str(config_path)], str(output_dir)
        )
      self.assertEqual(succeeded, 0)
      self.assertEqual(len(failed), 1)
      self.assertIn("dt_room_change=0.280 K", failed[0].reason)

  def test_export_sweep_csv_includes_sweep_params_and_non_safe_metrics(
      self,
  ) -> None:
    """Verifies CSV export has sweep param columns and non-*_safe metric columns."""
    point = run_sweep.SweepPoint(
        s_kiln_mm=200.0,
        w_floor_pad_mm=1200.0,
        t_floor_pad_mm=0.0,
        w_door_open_mm=150.0,
        exhaust_fan_cfm=400.0,
        d_reflector_mm=800.0,
        t_reflector_calsil_mm=10.0,
        g_floor_reflector_mm=10.0,
        d_chimney_mm=0.0,
        g_chimney_mm=0.0,
        w_shield_mm=1200.0,
        d_hood_inlet_mm=800.0,
        bung_captured=True,
    )
    cfg = run_sweep.build_config_proto(point)
    cfg.grid.dx_wall_mm = 200.0
    cfg.grid.dz_wall_mm = 200.0
    cfg.grid.n_theta_kiln = 8
    cfg.grid.n_z_kiln = 3

    with tempfile.TemporaryDirectory() as tmp_dir:
      tmp_path = pathlib.Path(tmp_dir)
      config_path = tmp_path / f"{point.experiment_name}.textproto"
      output_dir = tmp_path / "results"
      output_dir.mkdir()
      run_sweep.write_config_textproto(cfg, config_path)
      run_sweep._run_geometry_group([str(config_path)], str(output_dir))

      csv_path = tmp_path / "sweep_summary.csv"
      rows_written = run_sweep.export_sweep_csv(output_dir, csv_path)
      self.assertEqual(rows_written, 1)
      self.assertTrue(csv_path.is_file())

      lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
      self.assertEqual(len(lines), 2)
      headers = lines[0].split(",")

      expected_param_cols = [
          "s_kiln_mm",
          "w_floor_pad_mm",
          "t_floor_pad_mm",
          "w_door_open_mm",
          "exhaust_fan_cfm",
          "d_reflector_mm",
          "t_reflector_calsil_mm",
          "g_floor_reflector_mm",
          "d_chimney_mm",
          "g_chimney_mm",
          "w_shield_mm",
          "d_hood_inlet_mm",
          "bung_captured",
      ]
      for col in expected_param_cols:
        self.assertIn(col, headers)

      for col in headers:
        self.assertFalse(
            col.endswith("_safe"),
            f"Boolean *_safe metric column {col!r} should be excluded.",
        )

      self.assertIn("t_gypsum_max_left_c", headers)
      self.assertIn("plume_fully_captured", headers)
      self.assertIn("t_ceiling_max_c", headers)


if __name__ == "__main__":
  unittest.main()

