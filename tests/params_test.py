"""Unit tests for kiln_sim.params."""

import copy
import os
import pathlib

from absl.testing import absltest
from src import constants
from src import io_manager
from src import params as params_lib


def _find_baseline_config_path() -> pathlib.Path:
  """Locates configs/baseline_205mm.textproto in Bazel runfiles or repo root."""
  candidates = [
      pathlib.Path("configs/baseline_205mm.textproto"),
  ]
  test_srcdir = os.environ.get("TEST_SRCDIR")
  if test_srcdir:
    for root in pathlib.Path(test_srcdir).iterdir():
      candidates.append(root / "configs" / "baseline_205mm.textproto")
  for path in candidates:
    if path.is_file():
      return path
  raise FileNotFoundError("Could not locate configs/baseline_205mm.textproto")


class ParamsTest(absltest.TestCase):
  """Tests derived parameter geometry, vertical datums, and bounds validation."""

  def setUp(self) -> None:
    super().setUp()
    self.config_path = _find_baseline_config_path()
    self.base_pb, self.md5 = io_manager.load_simulation_config(self.config_path)

  def test_baseline_derived_geometry_and_vertical_stack(self) -> None:
    params = params_lib.from_proto(self.base_pb)

    self.assertAlmostEqual(
        params.h_shield,
        params.h_wall - params.g_floor - params.g_ceil,
        places=6,
    )
    self.assertAlmostEqual(
        params.x_shield_left_start,
        params.d_chimney + params.t_shield,
        places=6,
    )
    self.assertAlmostEqual(
        params.y_shield_back_start, params.d_chimney, places=6
    )

  def test_floor_pad_thickness_shifts_kiln_assembly_rigidly(self) -> None:
    """Increasing t_floor_pad_mm by +10 mm shifts all kiln heights by +10 mm."""
    p0 = params_lib.from_proto(self.base_pb)

    cfg_plus = copy.deepcopy(self.base_pb)
    cfg_plus.geometry.floor.t_floor_pad_mm += 10.0
    p1 = params_lib.from_proto(cfg_plus)

    dz = 0.010
    self.assertAlmostEqual(p1.z_pad_top - p0.z_pad_top, dz, places=6)
    self.assertAlmostEqual(
        p1.z_reflector_top - p0.z_reflector_top, dz, places=6
    )
    self.assertAlmostEqual(p1.z_drum_bot - p0.z_drum_bot, dz, places=6)
    self.assertAlmostEqual(p1.z_kiln_top - p0.z_kiln_top, dz, places=6)
    self.assertAlmostEqual(p1.z_ssr_bot - p0.z_ssr_bot, dz, places=6)
    self.assertAlmostEqual(p1.z_bung - p0.z_bung, dz, places=6)
    self.assertAlmostEqual(p1.z_hood_inlet, p0.z_hood_inlet, places=6)

  def test_reflector_elevation_is_independent_of_kiln_stand_height(self) -> None:
    """Changing g_floor_reflector_mm or t_reflector_calsil_mm does not shift the kiln."""
    p0 = params_lib.from_proto(self.base_pb)

    cfg_ref = copy.deepcopy(self.base_pb)
    cfg_ref.geometry.floor.g_floor_reflector_mm += 25.0
    cfg_ref.geometry.floor.t_reflector_calsil_mm += 10.0
    p1 = params_lib.from_proto(cfg_ref)

    self.assertAlmostEqual(
        p1.z_reflector_bot - p0.z_reflector_bot, 0.025, places=6
    )
    self.assertAlmostEqual(
        p1.z_reflector_top - p0.z_reflector_top, 0.035, places=6
    )
    self.assertAlmostEqual(p1.z_drum_bot, p0.z_drum_bot, places=6)
    self.assertAlmostEqual(p1.z_kiln_top, p0.z_kiln_top, places=6)
    self.assertAlmostEqual(p1.z_ssr_bot, p0.z_ssr_bot, places=6)
    self.assertAlmostEqual(p1.z_bung, p0.z_bung, places=6)

  def test_validation_rejects_reflector_at_or_above_kiln_bottom(self) -> None:
    """Rejects reflector configurations where z_reflector_top >= z_drum_bot."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.geometry.floor.g_floor_reflector_mm = cfg.geometry.kiln.h_kiln_base_mm
    cfg.geometry.floor.t_reflector_calsil_mm = 0.0
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg)

    cfg_above = copy.deepcopy(self.base_pb)
    cfg_above.geometry.floor.g_floor_reflector_mm = (
        cfg_above.geometry.kiln.h_kiln_base_mm - 10.0
    )
    cfg_above.geometry.floor.t_reflector_calsil_mm = 20.0
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg_above)

  def test_zero_chimney_and_zero_floor_ceiling_gaps_are_allowed(self) -> None:
    """Allows d_chimney_mm=0, g_floor_mm=0, and g_ceil_mm=0 for no-chimney runs."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.geometry.room.chimney.d_chimney_mm = 0.0
    cfg.geometry.shield.g_floor_mm = 0.0
    cfg.geometry.shield.g_ceil_mm = 0.0

    params = params_lib.from_proto(cfg)
    self.assertEqual(params.d_chimney, 0.0)
    self.assertEqual(params.d_chimney_eff, 0.0)
    self.assertEqual(params.g_floor, 0.0)
    self.assertEqual(params.g_ceil, 0.0)
    self.assertAlmostEqual(params.h_shield, params.h_wall, places=6)

  def test_ua_upper_zone_at_ceiling_equals_roof_only(self) -> None:
    """When stratification layer is at ceiling (z_layer = h_wall), only roof conducts."""
    params = params_lib.from_proto(self.base_pb)
    ua_at_ceiling = params.compute_ua_upper_zone(params.h_wall)
    expected_roof_ua = params.u_roof * params.l_room * params.w_room
    self.assertAlmostEqual(ua_at_ceiling, expected_roof_ua, places=6)

  def test_ua_upper_plus_lower_is_independent_of_z_layer(self) -> None:
    """Sum of upper and lower zone envelope UA equals full room envelope UA."""
    params = params_lib.from_proto(self.base_pb)
    total_at_floor = (
        params.compute_ua_upper_zone(0.0) + params.compute_ua_lower_zone(0.0)
    )
    for z in (0.5, 1.2, 1.8, params.h_wall):
      total_at_z = (
          params.compute_ua_upper_zone(z) + params.compute_ua_lower_zone(z)
      )
      self.assertAlmostEqual(total_at_z, total_at_floor, places=6)
      self.assertGreater(params.compute_ua_lower_zone(z), 0.0)

  def test_validation_rejects_negative_gaps(self) -> None:
    cfg = copy.deepcopy(self.base_pb)
    cfg.geometry.shield.g_floor_mm = (
        params_lib.MIN_FLOOR_GAP_M * params_lib.M_TO_MM - 1.0
    )
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg)

    cfg_chim = copy.deepcopy(self.base_pb)
    cfg_chim.geometry.room.chimney.d_chimney_mm = (
        params_lib.MIN_CHIMNEY_GAP_M * params_lib.M_TO_MM - 1.0
    )
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg_chim)

  def test_validation_rejects_insufficient_kiln_standoff(self) -> None:
    cfg = copy.deepcopy(self.base_pb)
    cfg.geometry.kiln.s_kiln_left_mm = (
        params_lib.MIN_KILN_STANDOFF_M * params_lib.M_TO_MM - 10.0
    )
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg)

  def test_validation_rejects_floor_pad_not_extending_under_all_sides_of_kiln(
      self,
  ) -> None:
    """Rejects configurations where the square floor pad does not cover the kiln."""
    p0 = params_lib.from_proto(self.base_pb)
    min_pad_x_mm = (p0.x_kiln_center + p0.r_kiln) * params_lib.M_TO_MM
    min_pad_y_mm = (p0.y_kiln_center + p0.r_kiln) * params_lib.M_TO_MM
    min_pad_mm = max(min_pad_x_mm, min_pad_y_mm)

    # 1. Pad slightly smaller than kiln outer edge in x/y raises ValueError
    cfg_small_pad = copy.deepcopy(self.base_pb)
    cfg_small_pad.geometry.floor.w_floor_pad_mm = min_pad_mm - 5.0
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg_small_pad)

    # 2. Moving kiln further from back wall so x_kiln_center + r_kiln > w_floor_pad
    overflow_x_mm = (
        p0.w_floor_pad * params_lib.M_TO_MM - min_pad_x_mm
    ) + 10.0
    cfg_shifted_x = copy.deepcopy(self.base_pb)
    cfg_shifted_x.geometry.kiln.s_kiln_back_mm += overflow_x_mm
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg_shifted_x)

    # 3. Moving kiln further from left wall so y_kiln_center + r_kiln > w_floor_pad
    overflow_y_mm = (p0.w_floor_pad * params_lib.M_TO_MM - min_pad_y_mm) + 10.0
    cfg_shifted_y = copy.deepcopy(self.base_pb)
    cfg_shifted_y.geometry.kiln.s_kiln_left_mm += overflow_y_mm
    with self.assertRaises(ValueError):
      params_lib.from_proto(cfg_shifted_y)

  def test_separate_kiln_emissivities_and_deprecated_eps_kiln_fallback(
      self,
  ) -> None:
    """Verifies Ecotop 60 baseline dimensions, emissivities, and eps_kiln fallback."""
    params = params_lib.from_proto(self.base_pb)
    self.assertAlmostEqual(params.eps_kiln_side, 0.25, places=6)
    self.assertAlmostEqual(params.eps_kiln_top, 0.85, places=6)
    self.assertAlmostEqual(params.eps_kiln_bot, 0.15, places=6)
    self.assertAlmostEqual(params.d_kiln, 0.635, places=6)
    self.assertAlmostEqual(params.h_drum, 0.531, places=6)
    self.assertAlmostEqual(params.t_lid, 0.076, places=6)
    self.assertAlmostEqual(params.h_kiln_base, 0.167, places=6)
    self.assertAlmostEqual(params.w_ssr, 0.148, places=6)
    self.assertAlmostEqual(params.d_ssr, 0.148, places=6)
    self.assertAlmostEqual(params.h_ssr, 0.362, places=6)
    self.assertAlmostEqual(params.u_kiln_side, 0.68, places=6)
    self.assertAlmostEqual(params.u_kiln_top, 1.20, places=6)
    self.assertAlmostEqual(params.u_kiln_bot, 0.50, places=6)

    cfg_legacy = copy.deepcopy(self.base_pb)
    cfg_legacy.thermal.emissivities.eps_kiln_side = 0.0
    cfg_legacy.thermal.emissivities.eps_kiln_top = 0.0
    cfg_legacy.thermal.emissivities.eps_kiln_bot = 0.0
    cfg_legacy.thermal.emissivities.eps_kiln = 0.45
    p_legacy = params_lib.from_proto(cfg_legacy)
    self.assertAlmostEqual(p_legacy.eps_kiln_side, 0.45, places=6)
    self.assertAlmostEqual(p_legacy.eps_kiln_top, 0.45, places=6)
    self.assertAlmostEqual(p_legacy.eps_kiln_bot, 0.45, places=6)

  def test_zero_reflector_diameter_is_valid(self) -> None:
    """Allows d_reflector_mm = 0.0 to simulate having no floor reflector."""
    cfg_no_ref = copy.deepcopy(self.base_pb)
    cfg_no_ref.geometry.floor.d_reflector_mm = 0.0
    cfg_no_ref.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_no_ref.geometry.floor.g_floor_reflector_mm = 0.0
    p_no_ref = params_lib.from_proto(cfg_no_ref)
    self.assertEqual(p_no_ref.d_reflector, 0.0)
    self.assertEqual(p_no_ref.r_reflector, 0.0)

  def test_open_lid_elevation_and_hood_clearance_warning(self) -> None:
    """Computes open-lid peak height and logs warning when colliding with hood."""
    cfg_low = copy.deepcopy(self.base_pb)
    cfg_low.geometry.floor.t_floor_pad_mm = 20.0
    cfg_low.geometry.kiln.h_kiln_base_mm = 167.0
    cfg_low.geometry.kiln.h_drum_mm = 531.0
    cfg_low.geometry.hood.z_hood_drop_mm = 1000.0
    p_low = params_lib.from_proto(cfg_low)
    expected_open_mm = (
        20.0 + 167.0 + 531.0 + constants.KILN_OPEN_LID_HEIGHT_ABOVE_RIM_MM
    )
    self.assertAlmostEqual(
        p_low.z_lid_open_top * params_lib.M_TO_MM, expected_open_mm, places=6
    )
    self.assertAlmostEqual(
        p_low.lid_open_hood_clearance * params_lib.M_TO_MM, 12.0, places=6
    )
    derived = p_low.to_derived_proto()
    self.assertAlmostEqual(derived.z_lid_open_top_mm, expected_open_mm, places=5)
    self.assertAlmostEqual(derived.lid_open_hood_clearance_mm, 12.0, places=5)

    cfg_high = copy.deepcopy(cfg_low)
    cfg_high.geometry.kiln.h_kiln_base_mm = 290.0
    with self.assertLogs(level="WARNING") as logs:
      p_high = params_lib.from_proto(cfg_high)
    self.assertAlmostEqual(
        p_high.lid_open_hood_clearance * params_lib.M_TO_MM, -111.0, places=6
    )
    self.assertTrue(
        any("Open kiln lid peak" in msg for msg in logs.output), logs.output
    )

  def test_kiln_standoff_measured_from_outer_drum_for_both_diameters(
      self,
  ) -> None:
    """Wall standoff s_kiln_* is measured from the cylindrical drum across presets."""
    for d_mm in (617.0, 635.0):
      cfg = copy.deepcopy(self.base_pb)
      cfg.geometry.kiln.d_kiln_mm = d_mm
      cfg.geometry.kiln.s_kiln_left_mm = 200.0
      cfg.geometry.kiln.s_kiln_back_mm = 200.0
      p = params_lib.from_proto(cfg)
      self.assertAlmostEqual(
          p.x_kiln_center - p.r_kiln - p.x_shield_front_back_wall,
          0.200,
          places=6,
      )
      self.assertAlmostEqual(
          p.y_kiln_center - p.r_kiln - p.y_shield_front_left_wall,
          0.200,
          places=6,
      )


if __name__ == "__main__":
  absltest.main()