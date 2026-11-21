"""End-to-end physics and equilibrium unit tests for kiln_sim.solver and metrics."""

import copy
import dataclasses
import math
import os
import pathlib
import tempfile
from unittest import mock

from absl.testing import absltest
from google.protobuf import text_format
import numpy as np
from src import constants
from src import io_manager
from src import mesh as mesh_lib
from src import metrics as metrics_lib
from src import params as params_lib
from src import solver as solver_lib
from src import view_factors as vf_lib
from proto import kiln_sim_pb2


def _find_baseline_config_path() -> pathlib.Path:
  candidates = [pathlib.Path("configs/baseline_205mm.textproto")]
  test_srcdir = os.environ.get("TEST_SRCDIR")
  if test_srcdir:
    for root in pathlib.Path(test_srcdir).iterdir():
      candidates.append(root / "configs" / "baseline_205mm.textproto")
  for path in candidates:
    if path.is_file():
      return path
  raise FileNotFoundError("Could not locate configs/baseline_205mm.textproto")


class SolverAndMetricsTest(absltest.TestCase):
  """Tests coupled thermal equilibrium, intuitiveness checks, and metrics."""

  def setUp(self) -> None:
    super().setUp()
    self.config_path = _find_baseline_config_path()
    self.base_pb, self.md5 = io_manager.load_simulation_config(
        self.config_path
    )
    self.base_pb.grid.dx_wall_mm = 200.0
    self.base_pb.grid.dz_wall_mm = 250.0
    self.base_pb.grid.n_theta_kiln = 12
    self.base_pb.grid.n_z_kiln = 4

  def test_kiln_turned_off_equilibrates_to_ambient_20c(self) -> None:
    """When kiln is off (T_internal = 20 C, Q_bung = 0, Q_ssr = 0), everything is 20 C."""
    t_amb_c = 20.0
    cfg_off = copy.deepcopy(self.base_pb)
    cfg_off.thermal.sources.t_internal_c = t_amb_c
    cfg_off.thermal.sources.t_ext_c = t_amb_c
    cfg_off.thermal.sources.q_bung_open_w = 0.0
    cfg_off.thermal.sources.q_ssr_internal_w = 0.0

    params = params_lib.from_proto(cfg_off)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    solution = solver_lib.solve_thermal_system(mesh, vf_sys, params)

    self.assertTrue(solution.converged)
    t_amb_k = t_amb_c + params_lib.KELVIN_OFFSET

    np.testing.assert_allclose(solution.t_front_k, t_amb_k, atol=0.05)
    np.testing.assert_allclose(solution.t_back_k, t_amb_k, atol=0.05)
    np.testing.assert_allclose(solution.q_rad_net_w_m2, 0.0, atol=0.5)
    np.testing.assert_allclose(solution.q_conv_net_w_m2, 0.0, atol=0.5)

    t_kiln_front_c = metrics_lib.get_kiln_front_face_temp_c(
        mesh=mesh, solution=solution, params=params
    )
    t_shield_behind_c = metrics_lib.get_shield_temp_behind_kiln_c(
        mesh=mesh, solution=solution, params=params
    )
    self.assertAlmostEqual(t_kiln_front_c, t_amb_c, delta=0.05)
    self.assertAlmostEqual(t_shield_behind_c, t_amb_c, delta=0.05)

    m = metrics_lib.compute_safety_metrics(
        mesh=mesh, vf_sys=vf_sys, solution=solution, params=params
    )
    self.assertAlmostEqual(m.t_vip_under_reflector_c, t_amb_c, delta=0.05)
    self.assertAlmostEqual(m.t_floor_pad_under_reflector_c, t_amb_c, delta=0.05)
    self.assertAlmostEqual(m.t_vip_outside_reflector_c, t_amb_c, delta=0.05)
    self.assertAlmostEqual(m.t_floor_pad_edge_c, t_amb_c, delta=0.05)
    self.assertAlmostEqual(m.t_vinyl_edge_c, t_amb_c, delta=0.05)
    self.assertAlmostEqual(m.t_ceiling_max_c, t_amb_c, delta=0.05)
    self.assertTrue(m.ceiling_safe)

  def test_baseline_firing_convergence_ordering_and_output_provenance(
      self,
  ) -> None:
    """Verifies physical temperature hierarchy and output serialization at 1280 C."""
    params = params_lib.from_proto(self.base_pb)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    solution = solver_lib.solve_thermal_system(mesh, vf_sys, params)

    self.assertTrue(solution.converged)
    self.assertLess(solution.max_energy_residual_w_m2, 0.5)

    t_kiln_front_c = metrics_lib.get_kiln_front_face_temp_c(
        mesh=mesh, solution=solution, params=params
    )
    t_shield_behind_c = metrics_lib.get_shield_temp_behind_kiln_c(
        mesh=mesh, solution=solution, params=params
    )

    output_pb = metrics_lib.build_simulation_output(
        config_pb=self.base_pb,
        input_config_path=str(self.config_path),
        input_config_md5=self.md5,
        timestamp_utc="2026-11-02T00:34:00Z",
        params=params,
        mesh=mesh,
        vf_sys=vf_sys,
        solution=solution,
    )
    m = output_pb.metrics

    self.assertBetween(t_kiln_front_c, 80.0, 180.0)
    self.assertGreater(t_kiln_front_c, t_shield_behind_c)
    self.assertGreater(t_shield_behind_c, m.t_aac_max_c)
    self.assertGreater(
        m.t_aac_max_c, self.base_pb.thermal.sources.t_ext_c
    )

    # Ceiling max temperature exceeds upper room air temperature due to
    # upward radiation from the kiln lid and uncaptured chimney convection.
    self.assertGreater(m.t_ceiling_max_c, m.t_room_upper_c)
    self.assertEqual(
        m.ceiling_safe, m.t_ceiling_max_c < metrics_lib.LIMIT_CEILING_C
    )

    self.assertLess(m.t_vip_under_reflector_c, m.t_vip_outside_reflector_c)
    self.assertTrue(m.floor_under_reflector_safe)
    self.assertEqual(
        m.floor_safe,
        m.floor_under_reflector_safe and m.floor_outside_reflector_safe,
    )

    self.assertBetween(m.t_gypsum_max_left_c, 22.0, 48.0)
    self.assertBetween(m.t_gypsum_max_back_c, 22.0, 48.0)
    self.assertGreater(m.lid_open_hood_clearance_mm, 0.0)
    self.assertTrue(m.lid_open_clearance_safe)
    self.assertEqual(
        m.gypsum_safe,
        (m.t_gypsum_max_left_c < metrics_lib.LIMIT_GYPSUM_C)
        and (m.t_gypsum_max_back_c < metrics_lib.LIMIT_GYPSUM_C),
    )

    self.assertAlmostEqual(
        m.t_gypsum_max_left_c, m.t_gypsum_max_back_c, delta=1.0
    )
    self.assertBetween(m.plume_capture_fraction, 0.90, 1.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
      out_path = io_manager.write_simulation_output(
          output_pb=output_pb,
          input_config_path=self.config_path,
          output_dir=pathlib.Path(tmp_dir),
          compact_timestamp="20261102T003400Z",
      )
      self.assertTrue(out_path.is_file())
      self.assertIn("20261102T003400Z.out.textproto", out_path.name)

      parsed_out = kiln_sim_pb2.SimulationOutput()
      text_format.Parse(out_path.read_text(encoding="utf-8"), parsed_out)
      self.assertEqual(parsed_out.provenance.input_config_md5, self.md5)
      self.assertEqual(
          parsed_out.input_config_snapshot.experiment_name,
          self.base_pb.experiment_name,
      )

  def test_front_mounted_ssr_receives_reflective_heating_from_polished_reflector(
      self,
  ) -> None:
    """An SSR mounted at +45 deg is heated by direct specular bounce off the reflector."""
    # Place the SSR box at +45 deg (pointing out into the open room, where its
    # faces do not look at the corner shields at x = 45 mm or y = 45 mm). With an
    # unshielded kiln bottom (u_kiln_bot = 2.0, eps_kiln_bot = 0.60), raising a
    # polished reflector (r = 500 mm, eps = 0.15) closer to the SSR bottom
    # (g_floor_reflector = 180 mm vs 15 mm) increases the specular bounce of
    # downward kiln-bottom radiation into the SSR casing and raises t_ssr_case_k.
    cfg_low = copy.deepcopy(self.base_pb)
    cfg_low.geometry.kiln.h_kiln_base_mm = 210.0
    cfg_low.thermal.sources.u_kiln_bot = 2.0
    cfg_low.thermal.emissivities.eps_kiln_bot = 0.60
    cfg_low.geometry.ssr.ssr_angle_deg = 45.0
    cfg_low.geometry.floor.d_reflector_mm = 1000.0
    cfg_low.geometry.floor.w_floor_pad_mm = 1200.0
    cfg_low.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_low.geometry.floor.g_floor_reflector_mm = 15.0
    p_low = params_lib.from_proto(cfg_low)
    m_low = mesh_lib.build_mesh(p_low)
    vf_low = vf_lib.compute_view_factors(m_low, p_low)
    sol_low = solver_lib.solve_thermal_system(m_low, vf_low, p_low)

    cfg_high = copy.deepcopy(cfg_low)
    cfg_high.geometry.floor.g_floor_reflector_mm = 180.0
    p_high = params_lib.from_proto(cfg_high)
    m_high = mesh_lib.build_mesh(p_high)
    vf_high = vf_lib.compute_view_factors(m_high, p_high)
    sol_high = solver_lib.solve_thermal_system(m_high, vf_high, p_high)

    self.assertGreater(sol_high.t_ssr_case_k, sol_low.t_ssr_case_k + 0.12)
    self.assertGreater(
        sol_high.t_ssr_case_k - sol_high.room.t_room_lower_k,
        sol_low.t_ssr_case_k - sol_low.room.t_room_lower_k + 0.25,
    )

  def test_wide_polished_reflector_increases_wall_and_ceiling_max_temps(
      self,
  ) -> None:
    """Widening a polished reflector beyond the kiln diameter heats walls and ceiling."""
    cfg_narrow = copy.deepcopy(self.base_pb)
    cfg_narrow.geometry.floor.d_reflector_mm = cfg_narrow.geometry.kiln.d_kiln_mm
    cfg_narrow.geometry.floor.w_floor_pad_mm = 1200.0
    cfg_narrow.thermal.emissivities.eps_reflector_top = 0.15
    p_narrow = params_lib.from_proto(cfg_narrow)
    mesh_narrow = mesh_lib.build_mesh(p_narrow)
    vf_narrow = vf_lib.compute_view_factors(mesh_narrow, p_narrow)
    sol_narrow = solver_lib.solve_thermal_system(
        mesh_narrow, vf_narrow, p_narrow
    )
    m_narrow = metrics_lib.compute_safety_metrics(
        mesh=mesh_narrow,
        vf_sys=vf_narrow,
        solution=sol_narrow,
        params=p_narrow,
    )

    cfg_wide = copy.deepcopy(cfg_narrow)
    cfg_wide.geometry.floor.d_reflector_mm = 1000.0  # r_ref = 500 mm > r_kiln
    p_wide = params_lib.from_proto(cfg_wide)
    mesh_wide = mesh_lib.build_mesh(p_wide)
    vf_wide = vf_lib.compute_view_factors(mesh_wide, p_wide)
    sol_wide = solver_lib.solve_thermal_system(mesh_wide, vf_wide, p_wide)
    m_wide = metrics_lib.compute_safety_metrics(
        mesh=mesh_wide,
        vf_sys=vf_wide,
        solution=sol_wide,
        params=p_wide,
    )

    self.assertGreater(m_wide.t_shield_max_c, m_narrow.t_shield_max_c)
    t_shield_behind_narrow = metrics_lib.get_shield_temp_behind_kiln_c(
        mesh=mesh_narrow, solution=sol_narrow, params=p_narrow
    )
    t_shield_behind_wide = metrics_lib.get_shield_temp_behind_kiln_c(
        mesh=mesh_wide, solution=sol_wide, params=p_wide
    )
    self.assertGreater(t_shield_behind_wide, t_shield_behind_narrow)
    self.assertGreater(
        m_wide.t_gypsum_max_left_c, m_narrow.t_gypsum_max_left_c
    )
    self.assertGreater(
        m_wide.t_gypsum_max_left_c - m_wide.t_room_upper_c,
        m_narrow.t_gypsum_max_left_c - m_narrow.t_room_upper_c,
    )
    self.assertGreater(
        m_wide.t_gypsum_max_back_c, m_narrow.t_gypsum_max_back_c
    )
    self.assertGreater(m_wide.t_ceiling_max_c, m_narrow.t_ceiling_max_c)
    self.assertGreater(
        m_wide.t_ceiling_max_c - m_wide.t_room_upper_c,
        m_narrow.t_ceiling_max_c - m_narrow.t_room_upper_c,
    )

    # Further widening an already overhanging reflector from 800 mm to 1100 mm
    # must continue to increase wall and ceiling peak temperatures.
    cfg_800 = copy.deepcopy(cfg_narrow)
    cfg_800.geometry.floor.d_reflector_mm = 800.0
    p_800 = params_lib.from_proto(cfg_800)
    m_800 = mesh_lib.build_mesh(p_800)
    vf_800 = vf_lib.compute_view_factors(m_800, p_800)
    sol_800 = solver_lib.solve_thermal_system(m_800, vf_800, p_800)
    metrics_800 = metrics_lib.compute_safety_metrics(
        mesh=m_800, vf_sys=vf_800, solution=sol_800, params=p_800
    )

    cfg_1100 = copy.deepcopy(cfg_narrow)
    cfg_1100.geometry.floor.d_reflector_mm = 1100.0
    p_1100 = params_lib.from_proto(cfg_1100)
    m_1100 = mesh_lib.build_mesh(p_1100)
    vf_1100 = vf_lib.compute_view_factors(m_1100, p_1100)
    sol_1100 = solver_lib.solve_thermal_system(m_1100, vf_1100, p_1100)
    metrics_1100 = metrics_lib.compute_safety_metrics(
        mesh=m_1100, vf_sys=vf_1100, solution=sol_1100, params=p_1100
    )

    self.assertGreater(metrics_1100.t_shield_max_c, metrics_800.t_shield_max_c)
    self.assertGreater(
        metrics_1100.t_gypsum_max_left_c, metrics_800.t_gypsum_max_left_c
    )
    self.assertGreater(
        metrics_1100.t_gypsum_max_back_c, metrics_800.t_gypsum_max_back_c
    )
    self.assertGreater(
        metrics_1100.t_ceiling_max_c, metrics_800.t_ceiling_max_c
    )

    # For the wide 1000 mm reflector, a polished top surface (eps = 0.15) must
    # also produce higher wall and ceiling peak temperatures than a matte/black
    # top surface (eps = 0.95) due to specular reflection of kiln radiation.
    cfg_wide_black = copy.deepcopy(cfg_wide)
    cfg_wide_black.thermal.emissivities.eps_reflector_top = 0.95
    p_wide_black = params_lib.from_proto(cfg_wide_black)
    vf_wide_black = vf_lib.compute_view_factors(mesh_wide, p_wide_black)
    sol_wide_black = solver_lib.solve_thermal_system(
        mesh_wide, vf_wide_black, p_wide_black
    )
    m_wide_black = metrics_lib.compute_safety_metrics(
        mesh=mesh_wide,
        vf_sys=vf_wide_black,
        solution=sol_wide_black,
        params=p_wide_black,
    )
    self.assertGreater(m_wide.t_shield_max_c, m_wide_black.t_shield_max_c)
    self.assertGreater(
        m_wide.t_gypsum_max_left_c, m_wide_black.t_gypsum_max_left_c
    )
    self.assertGreater(
        m_wide.t_gypsum_max_back_c, m_wide_black.t_gypsum_max_back_c
    )
    self.assertGreater(m_wide.t_ceiling_max_c, m_wide_black.t_ceiling_max_c)

    # Even when the kiln bottom is heavily insulated (so downward radiation onto
    # the overhanging reflector annulus comes primarily from the cylindrical
    # KILN_SIDE wall), widening a polished reflector from 700 mm to 1000 mm must
    # increase wall and ceiling peak temperatures.
    cfg_narrow_side = copy.deepcopy(cfg_narrow)
    cfg_narrow_side.thermal.sources.u_kiln_bot = 0.01
    p_ns = params_lib.from_proto(cfg_narrow_side)
    m_ns = mesh_lib.build_mesh(p_ns)
    vf_ns = vf_lib.compute_view_factors(m_ns, p_ns)
    sol_ns = solver_lib.solve_thermal_system(m_ns, vf_ns, p_ns)
    metrics_ns = metrics_lib.compute_safety_metrics(
        mesh=m_ns, vf_sys=vf_ns, solution=sol_ns, params=p_ns
    )

    cfg_wide_side = copy.deepcopy(cfg_wide)
    cfg_wide_side.thermal.sources.u_kiln_bot = 0.01
    p_ws = params_lib.from_proto(cfg_wide_side)
    m_ws = mesh_lib.build_mesh(p_ws)
    vf_ws = vf_lib.compute_view_factors(m_ws, p_ws)
    sol_ws = solver_lib.solve_thermal_system(m_ws, vf_ws, p_ws)
    metrics_ws = metrics_lib.compute_safety_metrics(
        mesh=m_ws, vf_sys=vf_ws, solution=sol_ws, params=p_ws
    )

    self.assertGreater(metrics_ws.t_shield_max_c, metrics_ns.t_shield_max_c)
    self.assertGreater(
        metrics_ws.t_gypsum_max_left_c, metrics_ns.t_gypsum_max_left_c
    )
    self.assertGreater(
        metrics_ws.t_gypsum_max_back_c, metrics_ns.t_gypsum_max_back_c
    )
    self.assertGreater(metrics_ws.t_ceiling_max_c, metrics_ns.t_ceiling_max_c)
    # Specifically, the radiative superheat of the ceiling above the upper room
    # gas temperature must increase due to kiln radiation reflecting off the
    # overhanging annulus up to the ceiling.
    self.assertGreater(
        metrics_ws.t_ceiling_max_c - metrics_ws.t_room_upper_c,
        metrics_ns.t_ceiling_max_c - metrics_ns.t_room_upper_c,
    )

  def test_floor_outside_reflector_peak_occurs_outside_reflector_shadow(
      self,
  ) -> None:
    """Peak floor/VIP temp outside an elevated overhanging reflector is outside its shadow."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.geometry.floor.d_reflector_mm = 760.0  # r_reflector = 380 mm > r_kiln (350 mm)
    cfg.geometry.floor.g_floor_reflector_mm = 80.0
    cfg.geometry.floor.t_reflector_calsil_mm = 20.0
    cfg.geometry.floor.w_floor_pad_mm = 1200.0
    params = params_lib.from_proto(cfg)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    sol = solver_lib.solve_thermal_system(mesh, vf_sys, params)
    metrics = metrics_lib.compute_safety_metrics(
        mesh=mesh, vf_sys=vf_sys, solution=sol, params=params
    )

    # Evaluate the floor pad temperature right at r = r_reflector + 5 mm (385 mm),
    # which sits inside the shadow cast by the elevated reflector rim.
    r_vip = params.t_vip / params.k_vip
    r_screed = params.t_screed / params.k_screed
    r_pad = (
        params.t_floor_pad / params.k_pad if params.t_floor_pad > 0.0 else 0.0
    )
    u_pad_subfloor = 1.0 / (r_pad + r_screed + r_vip)

    t_in_shadow_corner_k = metrics_lib._solve_floor_sensor_temp_k(
        vf_vec=vf_sys.sensor_vf["pad_edge_corner"],
        radiosity_w_m2=sol.radiosity_w_m2,
        eps_floor=metrics_lib.FLOOR_PAD_EMISSIVITY,
        u_subfloor=u_pad_subfloor,
        t_amb_k=sol.room.t_room_lower_k,
        t_ext_k=params.t_ext_k,
    )
    t_in_shadow_room_k = metrics_lib._solve_floor_sensor_temp_k(
        vf_vec=vf_sys.sensor_vf["pad_edge_room"],
        radiosity_w_m2=sol.radiosity_w_m2,
        eps_floor=metrics_lib.FLOOR_PAD_EMISSIVITY,
        u_subfloor=u_pad_subfloor,
        t_amb_k=sol.room.t_room_lower_k,
        t_ext_k=params.t_ext_k,
    )
    t_at_rim_plus_5mm_c = (
        max(t_in_shadow_corner_k, t_in_shadow_room_k) - params_lib.KELVIN_OFFSET
    )

    # Because points just outside the reflector's shadow have a direct view of
    # the hot kiln over the reflector rim, the reported peak floor pad
    # temperature outside the reflector must be strictly hotter than the
    # shadowed point at r_reflector + 5 mm.
    self.assertGreater(metrics.t_floor_pad_edge_c, t_at_rim_plus_5mm_c + 1.0)

  def test_unventilated_closed_door_zero_cfm_converges(self) -> None:
    """Unventilated (door=0, cfm=0) high-temperature room case converges."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.geometry.shield.w_shield_left_mm = 1200.0
    cfg.geometry.shield.w_shield_back_mm = 1200.0
    cfg.geometry.shield.g_floor_mm = 0.0
    cfg.geometry.shield.g_ceil_mm = 0.0
    cfg.geometry.shield.t_shield_mm = 12.5
    cfg.geometry.room.w_door_open_mm = 0.0
    cfg.geometry.room.chimney.d_chimney_mm = 0.0
    cfg.geometry.kiln.s_kiln_left_mm = 300.0
    cfg.geometry.kiln.s_kiln_back_mm = 300.0
    cfg.geometry.bung.bung_captured = True
    cfg.geometry.floor.d_reflector_mm = 1000.0
    cfg.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg.geometry.floor.g_floor_reflector_mm = 5.0
    cfg.geometry.floor.w_floor_pad_mm = 1400.0
    cfg.geometry.floor.t_floor_pad_mm = 0.0
    cfg.geometry.floor.t_screed_mm = 40.0
    cfg.geometry.hood.d_hood_inlet_mm = 600.0
    hood_center_mm = 12.5 + 300.0 + 0.5 * cfg.geometry.kiln.d_kiln_mm
    cfg.geometry.hood.x_hood_mm = hood_center_mm
    cfg.geometry.hood.y_hood_mm = hood_center_mm
    cfg.thermal.sources.exhaust_fan_cfm = 0.0

    params = params_lib.from_proto(cfg)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    sol = solver_lib.solve_thermal_system(mesh, vf_sys, params)
    metrics = metrics_lib.compute_safety_metrics(
        mesh=mesh, vf_sys=vf_sys, solution=sol, params=params
    )
    t_kiln_front_c = metrics_lib.get_kiln_front_face_temp_c(
        mesh=mesh, solution=sol, params=params
    )

    self.assertTrue(sol.converged)
    self.assertLess(sol.max_energy_residual_w_m2, 0.5)
    # In a sealed unventilated room (door=0, cfm=0), overturning plume
    # circulation heats the lower zone alongside the upper zone (no phantom 20C
    # outdoor air pinning t_room_lower_c near 22C), and the kiln outer skin must
    # remain strictly hotter than the room air and ceiling.
    self.assertGreater(metrics.t_room_lower_c, 60.0)
    self.assertLess(metrics.t_room_upper_c - metrics.t_room_lower_c, 25.0)
    self.assertGreater(t_kiln_front_c, metrics.t_room_upper_c)
    self.assertGreater(t_kiln_front_c, metrics.t_ceiling_max_c)

  def test_no_floor_reflector_and_foil_bottom_insulation_converge(self) -> None:
    """Simulating d_reflector_mm = 0 converges and foil bottom reduces VIP temp."""
    cfg_no_ref = copy.deepcopy(self.base_pb)
    cfg_no_ref.geometry.floor.d_reflector_mm = 0.0
    cfg_no_ref.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_no_ref.geometry.floor.g_floor_reflector_mm = 0.0
    p_foil = params_lib.from_proto(cfg_no_ref)
    m_foil = mesh_lib.build_mesh(p_foil)
    vf_foil = vf_lib.compute_view_factors(m_foil, p_foil)
    sol_foil = solver_lib.solve_thermal_system(m_foil, vf_foil, p_foil)
    metrics_foil = metrics_lib.compute_safety_metrics(
        mesh=m_foil, vf_sys=vf_foil, solution=sol_foil, params=p_foil
    )
    self.assertTrue(sol_foil.converged)
    self.assertLess(sol_foil.max_energy_residual_w_m2, 0.5)
    self.assertGreater(
        metrics_foil.t_vip_under_reflector_c,
        self.base_pb.thermal.sources.t_ext_c,
    )

    # Comparing foil-backed insulated bottom (u_kiln_bot=0.8, eps_kiln_bot=0.15)
    # against old uninsulated high-emissivity bottom (u_kiln_bot=2.0, eps=0.60)
    # confirms the foil-backed floor insulation plate significantly lowers floor
    # and VIP temperatures.
    cfg_old_bot = copy.deepcopy(cfg_no_ref)
    cfg_old_bot.thermal.sources.u_kiln_bot = 2.0
    cfg_old_bot.thermal.emissivities.eps_kiln_bot = 0.60
    p_old = params_lib.from_proto(cfg_old_bot)
    m_old = mesh_lib.build_mesh(p_old)
    vf_old = vf_lib.compute_view_factors(m_old, p_old)
    sol_old = solver_lib.solve_thermal_system(m_old, vf_old, p_old)
    metrics_old = metrics_lib.compute_safety_metrics(
        mesh=m_old, vf_sys=vf_old, solution=sol_old, params=p_old
    )
    self.assertLess(
        metrics_foil.t_vip_under_reflector_c + 15.0,
        metrics_old.t_vip_under_reflector_c,
    )
    self.assertLess(
        metrics_foil.t_vip_outside_reflector_c + 15.0,
        metrics_old.t_vip_outside_reflector_c,
    )

  def test_bare_stainless_reflector_uses_reflector_emissivity_on_underside(
      self,
  ) -> None:
    """Bare stainless reflector underside has low emissivity (eps_reflector_top)."""
    cfg_bare = copy.deepcopy(self.base_pb)
    cfg_bare.grid.dx_wall_mm = 200.0
    cfg_bare.grid.dz_wall_mm = 200.0
    cfg_bare.grid.n_theta_kiln = 8
    cfg_bare.grid.n_z_kiln = 3
    cfg_bare.geometry.floor.d_reflector_mm = 800.0
    cfg_bare.geometry.floor.w_floor_pad_mm = 1200.0
    cfg_bare.geometry.floor.g_floor_reflector_mm = 20.0
    cfg_bare.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_bare.thermal.sources.u_kiln_bot = 15.0
    cfg_bare.thermal.emissivities.eps_kiln_bot = 0.90
    cfg_bare.thermal.emissivities.eps_reflector_top = 0.15
    cfg_bare.thermal.emissivities.eps_shield = 0.90

    cfg_high_eps = copy.deepcopy(cfg_bare)
    # 0.5 mm layer triggers CalSil underside (eps_shield=0.90) with negligible R.
    cfg_high_eps.geometry.floor.t_reflector_calsil_mm = 0.5

    p_bare = params_lib.from_proto(cfg_bare)
    m_bare = mesh_lib.build_mesh(p_bare)
    vf_bare = vf_lib.compute_view_factors(m_bare, p_bare)
    sol_bare = solver_lib.solve_thermal_system(m_bare, vf_bare, p_bare)
    met_bare = metrics_lib.compute_safety_metrics(m_bare, vf_bare, sol_bare, p_bare)

    p_high = params_lib.from_proto(cfg_high_eps)
    m_high = mesh_lib.build_mesh(p_high)
    vf_high = vf_lib.compute_view_factors(m_high, p_high)
    sol_high = solver_lib.solve_thermal_system(m_high, vf_high, p_high)
    met_high = metrics_lib.compute_safety_metrics(m_high, vf_high, sol_high, p_high)

    self.assertLess(
        met_bare.t_floor_pad_under_reflector_c + 10.0,
        met_high.t_floor_pad_under_reflector_c,
    )

  def test_flush_reflector_conducts_to_subfloor_ext_sink(self) -> None:
    """When g_floor_reflector == 0, reflector conducts directly to t_ext_k."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.grid.dx_wall_mm = 200.0
    cfg.grid.dz_wall_mm = 200.0
    cfg.grid.n_theta_kiln = 8
    cfg.grid.n_z_kiln = 3
    cfg.geometry.floor.g_floor_reflector_mm = 0.0
    cfg.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg.geometry.floor.d_reflector_mm = 700.0
    cfg.geometry.floor.w_floor_pad_mm = 1200.0
    cfg.geometry.room.w_door_open_mm = 0.0
    cfg.thermal.sources.exhaust_fan_cfm = 0.0
    cfg.thermal.sources.t_ext_c = 10.0
    cfg.thermal.sources.u_kiln_bot = 0.01

    params = params_lib.from_proto(cfg)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    sol = solver_lib.solve_thermal_system(mesh, vf_sys, params)

    ref_mask = mesh.surface_types == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
    r_sub = (
        params.t_floor_pad / params.k_pad
        + params.t_screed / params.k_screed
        + params.t_vip / params.k_vip
    )
    u_sub = 1.0 / r_sub
    expected_q_down = u_sub * (float(sol.t_front_k[ref_mask][0]) - params.t_ext_k)
    actual_q_down = -float(sol.q_cond_net_w_m2[ref_mask][0])
    self.assertAlmostEqual(actual_q_down, expected_q_down, delta=1e-2)

  def test_coaxial_diffuse_and_specular_disks_match_parallel_plate_radiation(
      self,
  ) -> None:
    """Close coaxial KILN_BOTTOM and wide specular reflector match parallel plates."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.grid.dx_wall_mm = 250.0
    cfg.grid.dz_wall_mm = 250.0
    cfg.grid.n_theta_kiln = 8
    cfg.grid.n_z_kiln = 3
    cfg.geometry.kiln.d_kiln_mm = 700.0
    cfg.geometry.kiln.h_kiln_base_mm = 5.0
    cfg.geometry.floor.d_reflector_mm = 1400.0
    cfg.geometry.floor.w_floor_pad_mm = 1500.0
    cfg.geometry.floor.g_floor_reflector_mm = 0.0
    cfg.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg.thermal.emissivities.eps_kiln_bot = 0.80
    cfg.thermal.emissivities.eps_reflector_top = 0.15

    params = params_lib.from_proto(cfg)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    lu_piv = solver_lib.factor_radiosity_matrix(mesh, vf_sys)

    bot_id = int(
        np.flatnonzero(
            mesh.surface_types == int(mesh_lib.SurfaceType.KILN_BOTTOM)
        )[0]
    )
    t_front_k = np.full(mesh.num_patches, 300.0, dtype=np.float64)
    t_front_k[bot_id] = 700.0
    t_amb_rad_k = np.full(mesh.num_patches, 300.0, dtype=np.float64)

    _, q_rad_out = solver_lib.solve_radiosity_system(
        t_front_k=t_front_k,
        t_amb_rad_k=t_amb_rad_k,
        lu_piv=lu_piv,
        mesh=mesh,
        vf_sys=vf_sys,
    )
    q_parallel = (
        params_lib.STEFAN_BOLTZMANN
        * ((700.0 ** 4) - (300.0 ** 4))
        / ((1.0 / 0.80) + (1.0 / 0.15) - 1.0)
    )
    self.assertAlmostEqual(
        float(q_rad_out[bot_id]), q_parallel, delta=0.18 * q_parallel
    )

  def test_radiosity_system_conserves_global_watts_with_specular_reflector(
      self,
  ) -> None:
    """Specular reflector radiosity equals E_b and total enclosure Watts conserve."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.grid.dx_wall_mm = 250.0
    cfg.grid.dz_wall_mm = 250.0
    cfg.grid.n_theta_kiln = 8
    cfg.grid.n_z_kiln = 3
    cfg.geometry.floor.d_reflector_mm = 1000.0
    cfg.geometry.floor.w_floor_pad_mm = 1200.0
    cfg.thermal.emissivities.eps_reflector_top = 0.15

    params = params_lib.from_proto(cfg)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    lu_piv = solver_lib.factor_radiosity_matrix(mesh, vf_sys)

    stypes = mesh.surface_types
    ref_mask = stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
    ref_id = int(np.flatnonzero(ref_mask)[0])
    kiln_mask = (
        (stypes == int(mesh_lib.SurfaceType.KILN_SIDE))
        | (stypes == int(mesh_lib.SurfaceType.KILN_LID))
        | (stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
    )

    t_front_k = np.full(mesh.num_patches, 310.0, dtype=np.float64)
    t_front_k[kiln_mask] = 650.0
    t_front_k[ref_id] = 320.0
    t_amb_rad_k = np.full(mesh.num_patches, 295.0, dtype=np.float64)

    radiosity, q_rad_out = solver_lib.solve_radiosity_system(
        t_front_k=t_front_k,
        t_amb_rad_k=t_amb_rad_k,
        lu_piv=lu_piv,
        mesh=mesh,
        vf_sys=vf_sys,
    )
    sigma = params_lib.STEFAN_BOLTZMANN
    self.assertAlmostEqual(
        float(radiosity[ref_id]), sigma * (320.0 ** 4), places=6
    )

    eb_surf = sigma * (t_front_k ** 4)
    eb_room = sigma * (t_amb_rad_k ** 4)
    total_patch_w = float(np.sum(mesh.areas * q_rad_out))
    room_terms = np.where(
        ref_mask,
        mesh.areas
        * params.eps_reflector_top
        * vf_sys.f_room
        * (eb_surf - eb_room),
        mesh.areas * vf_sys.f_room * (radiosity - eb_room),
    )
    total_room_w = float(np.sum(room_terms))
    self.assertAlmostEqual(total_patch_w, total_room_w, delta=1e-5)

  def test_hot_ssr_and_reflector_convection_heats_sealed_room(self) -> None:
    """Internal SSR heat in a sealed zero-fan studio heats the lower room air."""
    cfg_ssr = copy.deepcopy(self.base_pb)
    cfg_ssr.thermal.sources.t_internal_c = 20.0
    cfg_ssr.thermal.sources.t_ext_c = 20.0
    cfg_ssr.thermal.sources.q_bung_open_w = 0.0
    cfg_ssr.thermal.sources.q_ssr_internal_w = 250.0
    cfg_ssr.geometry.room.w_door_open_mm = 0.0
    cfg_ssr.thermal.sources.exhaust_fan_cfm = 0.0

    params = params_lib.from_proto(cfg_ssr)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    sol = solver_lib.solve_thermal_system(mesh, vf_sys, params)

    self.assertTrue(sol.converged)
    self.assertGreater(sol.room.t_room_lower_k, params.t_ext_k + 3.0)

  def test_radiatively_heated_ceiling_sensor_is_stably_stratified_vs_floor_sensor(
      self,
  ) -> None:
    """A radiatively heated downward-facing ceiling sensor has lower h_c than a floor sensor."""
    vf_vec = np.array([0.25], dtype=np.float64)
    rad = np.array(
        [params_lib.STEFAN_BOLTZMANN * (650.0 ** 4)], dtype=np.float64
    )
    t_ceil_k = metrics_lib._solve_ceiling_sensor_temp_k(
        vf_vec, rad, 0.90, 0.20, 300.0, 293.15
    )
    t_floor_k = metrics_lib._solve_floor_sensor_temp_k(
        vf_vec, rad, 0.90, 0.20, 300.0, 293.15
    )
    self.assertGreater(t_ceil_k, t_floor_k + 3.0)

  def test_uncaptured_plume_spill_heats_ceiling_hood_rim_sensors(
      self,
  ) -> None:
    """Uncaptured warm plume gas spilling past the hood heats the ceiling rim."""
    params = params_lib.from_proto(self.base_pb)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    sol = solver_lib.solve_thermal_system(mesh, vf_sys, params)

    t_plume_hot = sol.room.t_room_upper_k + 60.0
    sol_captured = dataclasses.replace(
        sol,
        room=dataclasses.replace(
            sol.room,
            plume_capture_fraction=1.0,
            t_plume_at_hood_k=t_plume_hot,
        ),
    )
    sol_spill = dataclasses.replace(
        sol,
        room=dataclasses.replace(
            sol.room,
            plume_capture_fraction=0.2,
            t_plume_at_hood_k=t_plume_hot,
        ),
    )
    m_captured = metrics_lib.compute_safety_metrics(
        mesh=mesh, vf_sys=vf_sys, solution=sol_captured, params=params
    )
    m_spill = metrics_lib.compute_safety_metrics(
        mesh=mesh, vf_sys=vf_sys, solution=sol_spill, params=params
    )
    self.assertGreater(m_spill.t_ceiling_max_c, m_captured.t_ceiling_max_c + 2.0)

  def test_exposed_corner_aac_gap_raises_t_aac_max_c(self) -> None:
    """Exposing bare corner AAC via large g_ceil or g_floor raises peak AAC temp."""
    cfg_base = copy.deepcopy(self.base_pb)
    cfg_base.geometry.kiln.s_kiln_left_mm = 150.0
    cfg_base.geometry.kiln.s_kiln_back_mm = 150.0
    cfg_base.geometry.shield.g_floor_mm = 50.0
    cfg_base.geometry.shield.g_ceil_mm = 50.0
    p_base = params_lib.from_proto(cfg_base)
    m_base = mesh_lib.build_mesh(p_base)
    vf_base = vf_lib.compute_view_factors(m_base, p_base)
    sol_base = solver_lib.solve_thermal_system(m_base, vf_base, p_base)
    met_base = metrics_lib.compute_safety_metrics(
        mesh=m_base, vf_sys=vf_base, solution=sol_base, params=p_base
    )

    cfg_ceil_gap = copy.deepcopy(cfg_base)
    cfg_ceil_gap.geometry.shield.g_ceil_mm = 1850.0
    p_ceil_gap = params_lib.from_proto(cfg_ceil_gap)
    m_ceil_gap = mesh_lib.build_mesh(p_ceil_gap)
    vf_ceil_gap = vf_lib.compute_view_factors(m_ceil_gap, p_ceil_gap)
    sol_ceil_gap = solver_lib.solve_thermal_system(
        m_ceil_gap, vf_ceil_gap, p_ceil_gap
    )
    met_ceil_gap = metrics_lib.compute_safety_metrics(
        mesh=m_ceil_gap,
        vf_sys=vf_ceil_gap,
        solution=sol_ceil_gap,
        params=p_ceil_gap,
    )

    cfg_floor_gap = copy.deepcopy(cfg_base)
    cfg_floor_gap.geometry.shield.g_floor_mm = 800.0
    p_floor_gap = params_lib.from_proto(cfg_floor_gap)
    m_floor_gap = mesh_lib.build_mesh(p_floor_gap)
    vf_floor_gap = vf_lib.compute_view_factors(m_floor_gap, p_floor_gap)
    sol_floor_gap = solver_lib.solve_thermal_system(
        m_floor_gap, vf_floor_gap, p_floor_gap
    )
    met_floor_gap = metrics_lib.compute_safety_metrics(
        mesh=m_floor_gap,
        vf_sys=vf_floor_gap,
        solution=sol_floor_gap,
        params=p_floor_gap,
    )

    self.assertGreater(met_ceil_gap.t_aac_max_c, met_base.t_aac_max_c + 3.5)
    self.assertGreater(met_floor_gap.t_aac_max_c, met_base.t_aac_max_c + 3.5)

  def test_warm_start_matches_cold_start_and_satisfies_final_patch_residual(
      self,
  ) -> None:
    """Warm start matches cold start and final patches satisfy final room state."""
    cfg_seed = copy.deepcopy(self.base_pb)
    cfg_seed.geometry.room.w_door_open_mm = 0.0
    cfg_seed.thermal.sources.exhaust_fan_cfm = 0.0
    p_seed = params_lib.from_proto(cfg_seed)
    m_seed = mesh_lib.build_mesh(p_seed)
    vf_seed = vf_lib.compute_view_factors(m_seed, p_seed)
    sol_seed = solver_lib.solve_thermal_system(m_seed, vf_seed, p_seed)

    # Check final patch-to-room consistency at sol_seed.room.
    lu_seed = solver_lib.factor_radiosity_matrix(m_seed, vf_seed)
    t_amb_seed = np.where(
        m_seed.centroids[:, 2] >= sol_seed.room.z_stratification_layer_m,
        sol_seed.room.t_room_upper_k,
        sol_seed.room.t_room_lower_k,
    )
    _, q_rad_check = solver_lib.solve_radiosity_system(
        t_front_k=sol_seed.t_front_k,
        t_amb_rad_k=t_amb_seed,
        lu_piv=lu_seed,
        mesh=m_seed,
        vf_sys=vf_seed,
    )
    stypes = m_seed.surface_types
    char_lengths = np.full(m_seed.num_patches, p_seed.h_wall, dtype=np.float64)
    char_lengths[stypes == int(mesh_lib.SurfaceType.KILN_SIDE)] = (
        p_seed.h_kiln_total
    )
    char_lengths[
        (stypes == int(mesh_lib.SurfaceType.KILN_LID))
        | (stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
    ] = (0.5 * p_seed.r_kiln)
    char_lengths[stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)] = max(
        0.05, 0.5 * p_seed.r_reflector
    )
    char_lengths[
        (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT))
        | (stypes == int(mesh_lib.SurfaceType.SHIELD_BACK))
    ] = p_seed.h_shield
    char_lengths[stypes == int(mesh_lib.SurfaceType.SSR_BOX)] = p_seed.h_ssr
    h_check = solver_lib.compute_natural_convection_h(
        t_surf_k=sol_seed.t_front_k,
        t_amb_k=t_amb_seed,
        char_lengths_m=char_lengths,
        surface_types=stypes,
    )
    q_conv_check = h_check * (sol_seed.t_front_k - t_amb_seed)
    final_res_max = float(
        np.max(np.abs(q_rad_check + q_conv_check - sol_seed.q_cond_net_w_m2))
    )
    self.assertLess(final_res_max, 1e-2)

    p_base = params_lib.from_proto(self.base_pb)
    m_base = mesh_lib.build_mesh(p_base)
    vf_base = vf_lib.compute_view_factors(m_base, p_base)
    sol_cold = solver_lib.solve_thermal_system(m_base, vf_base, p_base)
    sol_warm = solver_lib.solve_thermal_system(
        m_base, vf_base, p_base, initial_solution=sol_seed
    )

    self.assertLess(
        float(np.max(np.abs(sol_warm.t_front_k - sol_cold.t_front_k))), 0.01
    )
    self.assertLess(
        abs(sol_warm.room.t_room_upper_k - sol_cold.room.t_room_upper_k), 0.01
    )
    self.assertLess(
        abs(
            sol_warm.room.z_stratification_layer_m
            - sol_cold.room.z_stratification_layer_m
        ),
        0.002,
    )

  def test_exhaust_duct_temp_includes_hood_enthalpy_and_teed_bung(self) -> None:
    """Exhaust duct temp equals ambient when off, rises when firing, and includes teed bung heat."""
    cfg_off = copy.deepcopy(self.base_pb)
    cfg_off.grid.dx_wall_mm = 200.0
    cfg_off.grid.dz_wall_mm = 200.0
    cfg_off.grid.n_theta_kiln = 8
    cfg_off.grid.n_z_kiln = 3
    cfg_off.thermal.sources.t_internal_c = 20.0
    cfg_off.thermal.sources.t_ext_c = 20.0
    cfg_off.thermal.sources.q_bung_open_w = 0.0
    cfg_off.thermal.sources.q_ssr_internal_w = 0.0
    cfg_off.thermal.sources.exhaust_fan_cfm = 400.0
    cfg_off.geometry.bung.bung_captured = True
    p_off = params_lib.from_proto(cfg_off)
    m_off = mesh_lib.build_mesh(p_off)
    vf_off = vf_lib.compute_view_factors(m_off, p_off)
    sol_off = solver_lib.solve_thermal_system(m_off, vf_off, p_off)
    met_off = metrics_lib.compute_safety_metrics(m_off, vf_off, sol_off, p_off)
    self.assertAlmostEqual(met_off.t_exhaust_duct_c, 20.0, delta=0.05)

    cfg_0 = copy.deepcopy(self.base_pb)
    cfg_0.grid.dx_wall_mm = 200.0
    cfg_0.grid.dz_wall_mm = 200.0
    cfg_0.grid.n_theta_kiln = 8
    cfg_0.grid.n_z_kiln = 3
    cfg_0.thermal.sources.exhaust_fan_cfm = 400.0
    cfg_0.geometry.room.w_door_open_mm = 150.0
    cfg_0.geometry.bung.bung_captured = True
    cfg_0.thermal.sources.q_bung_open_w = 0.0
    p_0 = params_lib.from_proto(cfg_0)
    m_0 = mesh_lib.build_mesh(p_0)
    vf_0 = vf_lib.compute_view_factors(m_0, p_0)
    sol_0 = solver_lib.solve_thermal_system(m_0, vf_0, p_0)
    met_0 = metrics_lib.compute_safety_metrics(m_0, vf_0, sol_0, p_0)
    self.assertGreater(met_0.t_exhaust_duct_c, met_0.t_room_lower_c + 1.0)

    cfg_250 = copy.deepcopy(cfg_0)
    cfg_250.thermal.sources.q_bung_open_w = 250.0
    p_250 = params_lib.from_proto(cfg_250)
    sol_250 = solver_lib.solve_thermal_system(m_0, vf_0, p_250)
    met_250 = metrics_lib.compute_safety_metrics(m_0, vf_0, sol_250, p_250)
    m_dot_hood = constants.AIR_RHO_REF * sol_0.room.v_dot_hood_eff_m3_s
    expected_dt = 250.0 / (m_dot_hood * constants.AIR_CP)
    self.assertAlmostEqual(
        met_250.t_exhaust_duct_c - met_0.t_exhaust_duct_c,
        expected_dt,
        delta=0.05,
    )

  def test_ceiling_max_temp_captures_radiative_peak_outside_hood_shadow(
      self,
  ) -> None:
    """t_ceiling_max_c finds peak heating outside the dropped hood's thermal shadow."""
    cfg = copy.deepcopy(self.base_pb)
    cfg.grid.dx_wall_mm = 200.0
    cfg.grid.dz_wall_mm = 200.0
    cfg.grid.n_theta_kiln = 8
    cfg.grid.n_z_kiln = 3
    cfg.geometry.room.chimney.d_chimney_mm = 0.0
    cfg.geometry.room.w_door_open_mm = 1500.0
    cfg.geometry.bung.bung_captured = True
    cfg.geometry.hood.d_hood_inlet_mm = 800.0
    cfg.geometry.hood.z_hood_drop_mm = 550.0
    cfg.thermal.sources.u_kiln_top = 5.0
    cfg.thermal.sources.exhaust_fan_cfm = 1500.0
    params = params_lib.from_proto(cfg)
    mesh = mesh_lib.build_mesh(params)
    vf_sys = vf_lib.compute_view_factors(mesh, params)
    sol = solver_lib.solve_thermal_system(mesh, vf_sys, params)
    metrics = metrics_lib.compute_safety_metrics(mesh, vf_sys, sol, params)

    legacy_names = (
        "ceiling_hood_rim_corner",
        "ceiling_hood_rim_room",
        "ceiling_chimney_corner",
    )
    t_legacy_max_c = (
        max(
            metrics_lib._solve_ceiling_sensor_temp_k(
                vf_vec=vf_sys.sensor_vf[name],
                radiosity_w_m2=sol.radiosity_w_m2,
                eps_ceil=params.eps_gypsum,
                u_roof=params.u_roof,
                t_amb_k=sol.room.t_room_upper_k,
                t_ext_k=params.t_ext_k,
            )
            for name in legacy_names
        )
        - constants.KELVIN_OFFSET
    )
    self.assertGreater(metrics.t_ceiling_max_c, t_legacy_max_c + 2.0)

  def test_ssr_kiln_facing_baffle_and_subkiln_air_heat_ssr_housing(
      self,
  ) -> None:
    """Sub-kiln convection and 4th kiln-facing wall baffle both heat the SSR."""
    cfg_base = copy.deepcopy(self.base_pb)
    cfg_base.grid.dx_wall_mm = 200.0
    cfg_base.grid.dz_wall_mm = 200.0
    cfg_base.grid.n_theta_kiln = 8
    cfg_base.grid.n_z_kiln = 3
    cfg_base.thermal.sources.q_ssr_internal_w = 0.0
    cfg_base.geometry.floor.d_reflector_mm = 0.0
    cfg_base.geometry.floor.g_floor_reflector_mm = 0.0
    cfg_base.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_base.geometry.room.w_door_open_mm = 1500.0
    cfg_base.thermal.sources.exhaust_fan_cfm = 450.0

    # 1. Pure sub-kiln convection (cool kiln side, low-eps hot kiln bottom):
    cfg_sub = copy.deepcopy(cfg_base)
    cfg_sub.thermal.sources.u_kiln_side = 0.01
    cfg_sub.thermal.emissivities.eps_kiln_side = 0.05
    cfg_sub.thermal.sources.u_kiln_bot = 2.5
    cfg_sub.thermal.emissivities.eps_kiln_bot = 0.05
    p_sub = params_lib.from_proto(cfg_sub)
    m_sub = mesh_lib.build_mesh(p_sub)
    vf_sub = vf_lib.compute_view_factors(m_sub, p_sub)
    sol_sub = solver_lib.solve_thermal_system(m_sub, vf_sub, p_sub)

    # 2. Thin-standoff SSR (d_ssr=20 mm) next to hot kiln jacket (no bottom heat):
    cfg_baf = copy.deepcopy(cfg_base)
    cfg_baf.geometry.ssr.d_ssr_mm = 20.0
    cfg_baf.thermal.sources.u_kiln_side = 2.5
    cfg_baf.thermal.sources.u_kiln_bot = 0.01
    cfg_baf.thermal.emissivities.eps_kiln_bot = 0.05
    p_baf = params_lib.from_proto(cfg_baf)
    m_baf = mesh_lib.build_mesh(p_baf)
    vf_baf = vf_lib.compute_view_factors(m_baf, p_baf)
    sol_baf = solver_lib.solve_thermal_system(m_baf, vf_baf, p_baf)

    self.assertGreater(sol_sub.t_ssr_air_k - sol_sub.room.t_room_lower_k, 1.0)
    self.assertGreater(sol_baf.t_ssr_case_k - sol_baf.room.t_room_lower_k, 5.0)
    self.assertGreater(sol_baf.t_ssr_air_k - sol_baf.room.t_room_lower_k, 2.5)

  def test_subfloor_2d_lateral_conduction_smooths_hotspot_and_warms_adjacent_vinyl(
      self,
  ) -> None:
    """2D screed/pad spreading reduces peak pad hotspots and warms adjacent vinyl."""
    cfg_a = copy.deepcopy(self.base_pb)
    cfg_a.grid.dx_wall_mm = 200.0
    cfg_a.grid.dz_wall_mm = 200.0
    cfg_a.grid.n_theta_kiln = 8
    cfg_a.grid.n_z_kiln = 3
    cfg_a.geometry.floor.d_reflector_mm = 730.0
    cfg_a.geometry.floor.g_floor_reflector_mm = 60.0
    cfg_a.thermal.sources.u_kiln_bot = 2.0
    cfg_a.thermal.emissivities.eps_kiln_bot = 0.80
    cfg_a.thermal.conductivities.k_pad = 2.0
    cfg_a.thermal.conductivities.k_screed = 1.8
    p_a = params_lib.from_proto(cfg_a)
    m_a = mesh_lib.build_mesh(p_a)
    vf_a = vf_lib.compute_view_factors(m_a, p_a)
    sol_a = solver_lib.solve_thermal_system(m_a, vf_a, p_a)
    met_a = metrics_lib.compute_safety_metrics(m_a, vf_a, sol_a, p_a)

    r_sub_a = (
        p_a.t_floor_pad / p_a.k_pad
        + p_a.t_screed / p_a.k_screed
        + p_a.t_vip / p_a.k_vip
    )
    pad_names = [
        name
        for name in vf_a.sensor_vf
        if name in ("pad_edge_corner", "pad_edge_room")
        or name.startswith("pad_outside_")
    ]
    t_1d_ring_peak_c = (
        max(
            metrics_lib._solve_floor_sensor_temp_k(
                vf_vec=vf_a.sensor_vf[name],
                radiosity_w_m2=sol_a.radiosity_w_m2,
                eps_floor=metrics_lib.FLOOR_PAD_EMISSIVITY,
                u_subfloor=1.0 / r_sub_a,
                t_amb_k=sol_a.room.t_room_lower_k,
                t_ext_k=p_a.t_ext_k,
            )
            for name in pad_names
        )
        - constants.KELVIN_OFFSET
    )
    self.assertLess(met_a.t_floor_pad_edge_c, t_1d_ring_peak_c - 1.5)

    cfg_b = copy.deepcopy(self.base_pb)
    cfg_b.grid.dx_wall_mm = 200.0
    cfg_b.grid.dz_wall_mm = 200.0
    cfg_b.grid.n_theta_kiln = 8
    cfg_b.grid.n_z_kiln = 3
    cfg_b.geometry.floor.d_reflector_mm = 720.0
    cfg_b.geometry.floor.g_floor_reflector_mm = 0.0
    cfg_b.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_b.thermal.emissivities.eps_reflector_top = 0.90
    cfg_b.geometry.floor.w_floor_pad_mm = 960.0
    cfg_b.geometry.kiln.h_kiln_base_mm = 40.0
    cfg_b.thermal.sources.u_kiln_bot = 6.0
    cfg_b.thermal.emissivities.eps_kiln_bot = 0.90
    cfg_b.thermal.sources.u_kiln_side = 0.10
    cfg_b.thermal.conductivities.k_pad = 2.0
    cfg_b.thermal.conductivities.k_screed = 2.0
    p_b = params_lib.from_proto(cfg_b)
    m_b = mesh_lib.build_mesh(p_b)
    vf_b = vf_lib.compute_view_factors(m_b, p_b)
    sol_b = solver_lib.solve_thermal_system(m_b, vf_b, p_b)
    met_b = metrics_lib.compute_safety_metrics(m_b, vf_b, sol_b, p_b)

    u_vinyl_b = 1.0 / (p_b.t_screed / p_b.k_screed + p_b.t_vip / p_b.k_vip)
    t_1d_vinyl_c = (
        max(
            metrics_lib._solve_floor_sensor_temp_k(
                vf_vec=vf_b.sensor_vf[name],
                radiosity_w_m2=sol_b.radiosity_w_m2,
                eps_floor=metrics_lib.VINYL_FLOOR_EMISSIVITY,
                u_subfloor=u_vinyl_b,
                t_amb_k=sol_b.room.t_room_lower_k,
                t_ext_k=p_b.t_ext_k,
            )
            for name in ("vinyl_edge_x", "vinyl_edge_y")
        )
        - constants.KELVIN_OFFSET
    )
    self.assertGreater(met_b.t_vinyl_edge_c, t_1d_vinyl_c + 5.0)

  def test_kiln_stand_leg_conductive_bridging_heats_subfloor_on_short_stand(
      self,
  ) -> None:
    """3 perimeter tripod legs with rubber feet bridge far less heat than bare steel."""
    cfg_tall = copy.deepcopy(self.base_pb)
    cfg_tall.grid.dx_wall_mm = 200.0
    cfg_tall.grid.dz_wall_mm = 200.0
    cfg_tall.grid.n_theta_kiln = 8
    cfg_tall.grid.n_z_kiln = 3
    cfg_tall.thermal.sources.u_kiln_bot = 0.05
    cfg_tall.thermal.emissivities.eps_kiln_bot = 0.05
    cfg_tall.thermal.sources.u_kiln_side = 1.2
    cfg_tall.geometry.floor.d_reflector_mm = 900.0
    cfg_tall.geometry.floor.t_reflector_calsil_mm = 20.0
    cfg_tall.geometry.floor.g_floor_reflector_mm = 5.0
    cfg_tall.geometry.floor.w_floor_pad_mm = 1100.0
    cfg_tall.geometry.kiln.h_kiln_base_mm = 250.0

    p_tall = params_lib.from_proto(cfg_tall)
    leg_pts = metrics_lib._kiln_leg_positions_m(p_tall)
    self.assertLen(leg_pts, 3)
    expected_r = p_tall.r_kiln + constants.KILN_LEG_RADIUS_OFFSET_M
    for lx, ly in leg_pts:
      self.assertAlmostEqual(
          math.hypot(lx - p_tall.x_kiln_center, ly - p_tall.y_kiln_center),
          expected_r,
          places=6,
      )

    cfg_short = copy.deepcopy(cfg_tall)
    cfg_short.geometry.kiln.h_kiln_base_mm = 35.0

    m_tall = mesh_lib.build_mesh(p_tall)
    vf_tall = vf_lib.compute_view_factors(m_tall, p_tall)
    sol_tall = solver_lib.solve_thermal_system(m_tall, vf_tall, p_tall)
    met_tall = metrics_lib.compute_safety_metrics(m_tall, vf_tall, sol_tall, p_tall)

    p_short = params_lib.from_proto(cfg_short)
    m_short = mesh_lib.build_mesh(p_short)
    vf_short = vf_lib.compute_view_factors(m_short, p_short)
    sol_short = solver_lib.solve_thermal_system(m_short, vf_short, p_short)
    met_short = metrics_lib.compute_safety_metrics(
        m_short, vf_short, sol_short, p_short
    )

    self.assertGreater(
        met_short.t_floor_pad_under_reflector_c,
        met_tall.t_floor_pad_under_reflector_c + 1.0,
    )
    self.assertGreater(
        met_short.t_vip_under_reflector_c,
        met_tall.t_vip_under_reflector_c + 0.5,
    )

    # Compare rubber-capped tripod stand against legacy bare-steel jacket legs.
    with (
        mock.patch.object(constants, "KILN_LEG_FOOT_CONTACT_R_K_W", 4.0),
        mock.patch.object(constants, "KILN_LEG_JACKET_TEMP_WEIGHT", 1.0),
    ):
      met_bare = metrics_lib.compute_safety_metrics(
          m_short, vf_short, sol_short, p_short
      )
    self.assertGreater(
        met_bare.t_floor_pad_under_reflector_c,
        met_short.t_floor_pad_under_reflector_c + 8.0,
    )

  def test_ssr_external_heatsink_reduces_internal_air_rise_and_recovers_trapped_limit(
      self,
  ) -> None:
    """External finned heatsink keeps SSR box air cooler and recovers trapped limit."""
    params = params_lib.from_proto(self.base_pb)
    t_amb_k = params.t_ext_k
    ssr_faces = np.full(3, t_amb_k, dtype=np.float64)
    ssr_areas = np.array([0.148 * 0.362] * 3, dtype=np.float64)

    # 1. Isothermal limit with q_ssr_internal_w == 0 equilibrates cleanly to ambient.
    p_zero = dataclasses.replace(params, q_ssr_internal_w=0.0)
    t_air_0, t_case_0 = solver_lib._solve_ssr_internal_air(
        t_ssr_faces_k=ssr_faces,
        ssr_areas=ssr_areas,
        t_ref_k=t_amb_k,
        t_lower_k=t_amb_k,
        params=p_zero,
        t_kiln_facing_k=t_amb_k,
        q_bot_conv_w=0.0,
    )
    self.assertAlmostEqual(t_air_0, t_amb_k, delta=1e-4)
    self.assertAlmostEqual(t_case_0, t_amb_k, delta=1e-4)

    # 2. External finned heatsink rejects heat to ambient vs. 100% trapped internal heat.
    t_air_hs, _ = solver_lib._solve_ssr_internal_air(
        t_ssr_faces_k=ssr_faces,
        ssr_areas=ssr_areas,
        t_ref_k=t_amb_k,
        t_lower_k=t_amb_k,
        params=params,
        t_kiln_facing_k=t_amb_k,
        q_bot_conv_w=0.0,
    )
    with (
        mock.patch.object(constants, "SSR_HEATSINK_POWER_FRACTION", 0.0),
        mock.patch.object(constants, "SSR_HEATSINK_UA_INT_W_K", 0.0),
    ):
      t_air_trapped, _ = solver_lib._solve_ssr_internal_air(
          t_ssr_faces_k=ssr_faces,
          ssr_areas=ssr_areas,
          t_ref_k=t_amb_k,
          t_lower_k=t_amb_k,
          params=params,
          t_kiln_facing_k=t_amb_k,
          q_bot_conv_w=0.0,
      )
    with mock.patch.object(constants, "SSR_HEATSINK_R_TH_AMB_K_W", 1e12):
      t_air_insulated_hs, _ = solver_lib._solve_ssr_internal_air(
          t_ssr_faces_k=ssr_faces,
          ssr_areas=ssr_areas,
          t_ref_k=t_amb_k,
          t_lower_k=t_amb_k,
          params=params,
          t_kiln_facing_k=t_amb_k,
          q_bot_conv_w=0.0,
      )
    rise_hs = t_air_hs - t_amb_k
    rise_trapped = t_air_trapped - t_amb_k
    self.assertGreater(rise_hs, 0.2)
    self.assertLess(rise_hs * 2.0, rise_trapped)
    self.assertAlmostEqual(t_air_insulated_hs, t_air_trapped, delta=0.05)

  def test_polished_stainless_emissivity_raises_skin_temp_and_shifts_to_convection(
      self,
  ) -> None:
    """Lowering eps_kiln_side from 0.60 to 0.25 raises skin temp & plume convection."""
    cfg_dull = copy.deepcopy(self.base_pb)
    cfg_dull.thermal.sources.u_kiln_side = 0.75
    cfg_dull.thermal.sources.u_kiln_top = 1.70
    cfg_dull.thermal.sources.u_kiln_bot = 0.55
    cfg_dull.thermal.emissivities.eps_kiln_side = 0.60
    p_dull = params_lib.from_proto(cfg_dull)
    m_dull = mesh_lib.build_mesh(p_dull)
    vf_dull = vf_lib.compute_view_factors(m_dull, p_dull)
    sol_dull = solver_lib.solve_thermal_system(m_dull, vf_dull, p_dull)
    met_dull = metrics_lib.compute_safety_metrics(
        m_dull, vf_dull, sol_dull, p_dull
    )
    t_skin_dull = metrics_lib.get_kiln_front_face_temp_c(
        m_dull, sol_dull, p_dull
    )

    cfg_pol = copy.deepcopy(cfg_dull)
    cfg_pol.thermal.emissivities.eps_kiln_side = 0.25
    p_pol = params_lib.from_proto(cfg_pol)
    m_pol = mesh_lib.build_mesh(p_pol)
    vf_pol = vf_lib.compute_view_factors(m_pol, p_pol)
    sol_pol = solver_lib.solve_thermal_system(m_pol, vf_pol, p_pol)
    met_pol = metrics_lib.compute_safety_metrics(m_pol, vf_pol, sol_pol, p_pol)
    t_skin_pol = metrics_lib.get_kiln_front_face_temp_c(m_pol, sol_pol, p_pol)

    self.assertGreater(t_skin_pol, t_skin_dull + 15.0)
    self.assertLess(met_pol.t_shield_max_c, met_dull.t_shield_max_c)
    self.assertGreater(sol_pol.room.q_plume_conv_w, sol_dull.room.q_plume_conv_w)

  def test_open_lid_clearance_metric_flags_collision_on_high_stand_deep_hood(
      self,
  ) -> None:
    """SafetyMetrics flags open-lid hood collision on high stand vs low stand."""
    cfg_low = copy.deepcopy(self.base_pb)
    cfg_low.geometry.floor.t_floor_pad_mm = 20.0
    cfg_low.geometry.kiln.h_kiln_base_mm = 167.0
    cfg_low.geometry.hood.z_hood_drop_mm = 1000.0
    p_low = params_lib.from_proto(cfg_low)
    m_low = mesh_lib.build_mesh(p_low)
    vf_low = vf_lib.compute_view_factors(m_low, p_low)
    sol_low = solver_lib.solve_thermal_system(m_low, vf_low, p_low)
    met_low = metrics_lib.compute_safety_metrics(m_low, vf_low, sol_low, p_low)
    self.assertAlmostEqual(met_low.lid_open_hood_clearance_mm, 12.0, places=4)
    self.assertTrue(met_low.lid_open_clearance_safe)

    cfg_high = copy.deepcopy(cfg_low)
    cfg_high.geometry.kiln.h_kiln_base_mm = 290.0
    p_high = params_lib.from_proto(cfg_high)
    m_high = mesh_lib.build_mesh(p_high)
    vf_high = vf_lib.compute_view_factors(m_high, p_high)
    sol_high = solver_lib.solve_thermal_system(m_high, vf_high, p_high)
    met_high = metrics_lib.compute_safety_metrics(
        m_high, vf_high, sol_high, p_high
    )
    self.assertAlmostEqual(
        met_high.lid_open_hood_clearance_mm, -111.0, places=4
    )
    self.assertFalse(met_high.lid_open_clearance_safe)


if __name__ == "__main__":
  absltest.main()