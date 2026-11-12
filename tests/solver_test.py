"""End-to-end physics and equilibrium unit tests for kiln_sim.solver and metrics."""

import copy
import os
import pathlib
import tempfile

from absl.testing import absltest
from google.protobuf import text_format
import numpy as np
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

    self.assertBetween(t_kiln_front_c, 95.0, 160.0)
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

    self.assertBetween(m.t_gypsum_max_left_c, 35.0, metrics_lib.LIMIT_GYPSUM_C)
    self.assertBetween(m.t_gypsum_max_back_c, 35.0, metrics_lib.LIMIT_GYPSUM_C)
    self.assertTrue(m.gypsum_safe)

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

    self.assertGreater(sol_high.t_ssr_case_k, sol_low.t_ssr_case_k + 0.5)

  def test_wide_polished_reflector_increases_wall_and_ceiling_max_temps(
      self,
  ) -> None:
    """Widening a polished reflector beyond the kiln diameter heats walls and ceiling."""
    cfg_narrow = copy.deepcopy(self.base_pb)
    cfg_narrow.geometry.floor.d_reflector_mm = 700.0  # r_ref == r_kiln = 350 mm
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
    cfg.geometry.hood.x_hood_mm = 662.5
    cfg.geometry.hood.y_hood_mm = 662.5
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


if __name__ == "__main__":
  absltest.main()