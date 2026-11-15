"""Unit tests for kiln_sim.plume_room."""

import copy
import os
import pathlib

from absl.testing import absltest
import numpy as np
from src import chimney as chimney_lib
from src import io_manager
from src import mesh as mesh_lib
from src import params as params_lib
from src import plume_room as plume_room_lib


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


class PlumeRoomTest(absltest.TestCase):
  """Tests corner plume scaling, geometric capture continuity, and room balance."""

  def setUp(self) -> None:
    super().setUp()
    self.base_pb, _ = io_manager.load_simulation_config(
        _find_baseline_config_path()
    )
    self.base_pb.grid.dx_wall_mm = 200.0
    self.base_pb.grid.dz_wall_mm = 250.0
    self.base_pb.grid.n_theta_kiln = 12
    self.base_pb.grid.n_z_kiln = 4
    self.base_pb.geometry.room.chimney.d_chimney_mm = 0.0
    self.base_pb.geometry.bung.bung_captured = True

  def _make_plume_inputs(self, cfg, q_kiln_flux: float = 150.0):
    params = params_lib.from_proto(cfg)
    mesh = mesh_lib.build_mesh(params)
    t_front_k = np.full(mesh.num_patches, params.t_ext_k, dtype=np.float64)
    q_conv_front_w_m2 = np.zeros(mesh.num_patches, dtype=np.float64)
    kiln_mask = (mesh.surface_types == int(mesh_lib.SurfaceType.KILN_SIDE)) | (
        mesh.surface_types == int(mesh_lib.SurfaceType.KILN_LID)
    )
    q_conv_front_w_m2[kiln_mask] = q_kiln_flux
    chimney = chimney_lib.solve_chimney(
        t_shield_back_k=t_front_k,
        t_inlet_k=params.t_ext_k,
        mesh=mesh,
        params=params,
    )
    return params, mesh, t_front_k, q_conv_front_w_m2, chimney

  def test_corner_wall_restriction_reduces_plume_entrainment_at_fixed_heat_release(
      self,
  ) -> None:
    """Tight corner walls restrict air entrainment and produce a smaller, hotter plume."""
    cfg_corner = copy.deepcopy(self.base_pb)
    cfg_corner.geometry.room.h_wall_mm = 4500.0
    cfg_corner.geometry.hood.z_hood_drop_mm = 0.0
    cfg_corner.geometry.kiln.s_kiln_left_mm = 105.0
    cfg_corner.geometry.kiln.s_kiln_back_mm = 105.0
    p_tmp = params_lib.from_proto(cfg_corner)
    cfg_corner.geometry.hood.x_hood_mm = p_tmp.x_kiln_center * 1e3
    cfg_corner.geometry.hood.y_hood_mm = p_tmp.y_kiln_center * 1e3
    p_c, m_c, tf_c, qc_c, ch_c = self._make_plume_inputs(
        cfg_corner, q_kiln_flux=10000.0
    )
    sol_corner = plume_room_lib.solve_plume_and_room(
        mesh=m_c,
        t_front_k=tf_c,
        q_conv_front_w_m2=qc_c,
        chimney=ch_c,
        t_lower_prev_k=p_c.t_ext_k,
        params=p_c,
    )

    cfg_open = copy.deepcopy(cfg_corner)
    cfg_open.geometry.kiln.s_kiln_left_mm = 1500.0
    cfg_open.geometry.kiln.s_kiln_back_mm = 1500.0
    cfg_open.geometry.floor.w_floor_pad_mm = 2600.0
    p_o_tmp = params_lib.from_proto(cfg_open)
    cfg_open.geometry.hood.x_hood_mm = p_o_tmp.x_kiln_center * 1e3
    cfg_open.geometry.hood.y_hood_mm = p_o_tmp.y_kiln_center * 1e3
    p_o, m_o, tf_o, qc_o, ch_o = self._make_plume_inputs(
        cfg_open, q_kiln_flux=10000.0
    )
    sol_open = plume_room_lib.solve_plume_and_room(
        mesh=m_o,
        t_front_k=tf_o,
        q_conv_front_w_m2=qc_o,
        chimney=ch_o,
        t_lower_prev_k=p_o.t_ext_k,
        params=p_o,
    )

    self.assertGreater(sol_corner.m_eff_corner, 2.0)
    self.assertLess(sol_open.m_eff_corner, 1.05)
    self.assertAlmostEqual(
        sol_corner.q_plume_conv_w, sol_open.q_plume_conv_w, delta=1.0
    )
    self.assertLess(
        sol_corner.v_dot_plume_at_hood_m3_s,
        0.75 * sol_open.v_dot_plume_at_hood_m3_s,
    )
    self.assertGreater(
        sol_corner.t_plume_at_hood_k, sol_open.t_plume_at_hood_k + 5.0
    )

  def test_geometric_capture_fraction_is_continuous_across_r_plume_95(
      self,
  ) -> None:
    """Capture fraction is continuous at r_hood_inlet == r_plume_95 (no 5% jump)."""
    cfg0 = copy.deepcopy(self.base_pb)
    cfg0.thermal.sources.exhaust_fan_cfm = 2000.0
    cfg0.geometry.room.w_door_open_mm = 2000.0
    p0, m0, tf0, qc0, ch0 = self._make_plume_inputs(cfg0)
    sol0 = plume_room_lib.solve_plume_and_room(
        mesh=m0,
        t_front_k=tf0,
        q_conv_front_w_m2=qc0,
        chimney=ch0,
        t_lower_prev_k=p0.t_ext_k,
        params=p0,
    )
    r_95_mm = sol0.r_plume_at_hood_m * 1e3

    cfg_below = copy.deepcopy(cfg0)
    cfg_below.geometry.hood.d_hood_inlet_mm = 2.0 * (r_95_mm - 0.1)
    p_b, m_b, tf_b, qc_b, ch_b = self._make_plume_inputs(cfg_below)
    sol_below = plume_room_lib.solve_plume_and_room(
        mesh=m_b,
        t_front_k=tf_b,
        q_conv_front_w_m2=qc_b,
        chimney=ch_b,
        t_lower_prev_k=p_b.t_ext_k,
        params=p_b,
    )

    cfg_above = copy.deepcopy(cfg0)
    cfg_above.geometry.hood.d_hood_inlet_mm = 2.0 * (r_95_mm + 0.1)
    p_a, m_a, tf_a, qc_a, ch_a = self._make_plume_inputs(cfg_above)
    sol_above = plume_room_lib.solve_plume_and_room(
        mesh=m_a,
        t_front_k=tf_a,
        q_conv_front_w_m2=qc_a,
        chimney=ch_a,
        t_lower_prev_k=p_a.t_ext_k,
        params=p_a,
    )

    self.assertAlmostEqual(
        sol_below.plume_capture_fraction,
        sol_above.plume_capture_fraction,
        delta=1e-3,
    )

  def test_closed_door_throttles_effective_hood_exhaust_flow(self) -> None:
    """Closing the room door starves makeup air and throttles hood exhaust flow."""
    cfg_open = copy.deepcopy(self.base_pb)
    cfg_open.geometry.room.w_door_open_mm = 900.0
    cfg_open.thermal.sources.exhaust_fan_cfm = 400.0
    p_o, m_o, tf_o, qc_o, ch_o = self._make_plume_inputs(
        cfg_open, q_kiln_flux=400.0
    )
    sol_open = plume_room_lib.solve_plume_and_room(
        mesh=m_o,
        t_front_k=tf_o,
        q_conv_front_w_m2=qc_o,
        chimney=ch_o,
        t_lower_prev_k=p_o.t_ext_k,
        params=p_o,
    )

    cfg_closed = copy.deepcopy(self.base_pb)
    cfg_closed.geometry.room.w_door_open_mm = 0.0
    cfg_closed.thermal.sources.exhaust_fan_cfm = 400.0
    p_c, m_c, tf_c, qc_c, ch_c = self._make_plume_inputs(
        cfg_closed, q_kiln_flux=400.0
    )
    sol_closed = plume_room_lib.solve_plume_and_room(
        mesh=m_c,
        t_front_k=tf_c,
        q_conv_front_w_m2=qc_c,
        chimney=ch_c,
        t_lower_prev_k=p_c.t_ext_k,
        params=p_c,
    )

    self.assertGreater(sol_open.v_dot_hood_eff_m3_s, 0.95 * p_o.v_dot_hood_m3_s)
    self.assertGreater(sol_closed.v_dot_hood_eff_m3_s, 0.0)
    self.assertLess(sol_closed.v_dot_hood_eff_m3_s, 0.6 * p_c.v_dot_hood_m3_s)

  def test_two_zone_steady_energy_balance_closes_globally(self) -> None:
    """Room energy inputs (convection, radiation, louvers) equal envelope + airflow losses."""
    from src import constants

    for w_door_mm, fan_cfm in [(800.0, 250.0), (0.0, 400.0)]:
      cfg = copy.deepcopy(self.base_pb)
      cfg.geometry.room.w_door_open_mm = w_door_mm
      cfg.thermal.sources.exhaust_fan_cfm = fan_cfm
      p, m, tf, qc, ch = self._make_plume_inputs(cfg, q_kiln_flux=250.0)

      stypes = m.surface_types
      kiln_mask = (stypes == int(mesh_lib.SurfaceType.KILN_SIDE)) | (
          stypes == int(mesh_lib.SurfaceType.KILN_LID)
      )
      shield_mask = (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT)) | (
          stypes == int(mesh_lib.SurfaceType.SHIELD_BACK)
      )
      gyp_mask = (stypes == int(mesh_lib.SurfaceType.GYPSUM_LEFT)) | (
          stypes == int(mesh_lib.SurfaceType.GYPSUM_BACK)
      )
      qc[shield_mask] = 25.0
      qc[gyp_mask] = -12.0

      q_rad_room_patch_w = np.zeros(m.num_patches, dtype=np.float64)
      q_rad_room_patch_w[stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)] = (
          60.0
      )
      q_rad_room_patch_w[gyp_mask] = -2.0
      q_ssr_louver_w = 180.0
      q_under_ref_conv_w = 45.0

      sol = plume_room_lib.solve_plume_and_room(
          mesh=m,
          t_front_k=tf,
          q_conv_front_w_m2=qc,
          chimney=ch,
          t_lower_prev_k=p.t_ext_k,
          params=p,
          q_rad_room_patch_w=q_rad_room_patch_w,
          q_ssr_louver_w=q_ssr_louver_w,
          q_under_ref_conv_w=q_under_ref_conv_w,
      )

      non_plume_conv_w = np.where(
          kiln_mask,
          np.minimum(0.0, qc) * m.areas,
          qc * m.areas,
      )
      q_in = (
          sol.q_plume_conv_w
          + ch.q_conv_total_w
          + float(np.sum(non_plume_conv_w))
          + float(np.sum(q_rad_room_patch_w))
          + q_ssr_louver_w
          + q_under_ref_conv_w
      )
      ua_upper = p.compute_ua_upper_zone(sol.z_stratification_layer_m)
      ua_lower = p.compute_ua_lower_zone(sol.z_stratification_layer_m)
      q_out = (
          ua_upper * (sol.t_room_upper_k - p.t_ext_k)
          + ua_lower * (sol.t_room_lower_k - p.t_ext_k)
          + sol.m_dot_door_buoy_kg_s
          * constants.AIR_CP
          * (sol.t_room_upper_k - p.t_ext_k)
          + sol.q_hood_exhaust_w
      )
      self.assertAlmostEqual(q_in, q_out, delta=1e-4)

  def test_undersized_exhaust_duct_throttles_effective_blower_flow(
      self,
  ) -> None:
    """Narrowing the exhaust duct from 200 mm to 100 mm throttles blower flow."""
    cfg_200 = copy.deepcopy(self.base_pb)
    cfg_200.geometry.room.w_door_open_mm = 900.0
    cfg_200.geometry.hood.d_hood_duct_mm = 200.0
    cfg_200.thermal.sources.exhaust_fan_cfm = 400.0
    p_200, m_200, tf_200, qc_200, ch_200 = self._make_plume_inputs(
        cfg_200, q_kiln_flux=400.0
    )
    sol_200 = plume_room_lib.solve_plume_and_room(
        mesh=m_200,
        t_front_k=tf_200,
        q_conv_front_w_m2=qc_200,
        chimney=ch_200,
        t_lower_prev_k=p_200.t_ext_k,
        params=p_200,
    )

    cfg_100 = copy.deepcopy(cfg_200)
    cfg_100.geometry.hood.d_hood_duct_mm = 100.0
    p_100, m_100, tf_100, qc_100, ch_100 = self._make_plume_inputs(
        cfg_100, q_kiln_flux=400.0
    )
    sol_100 = plume_room_lib.solve_plume_and_room(
        mesh=m_100,
        t_front_k=tf_100,
        q_conv_front_w_m2=qc_100,
        chimney=ch_100,
        t_lower_prev_k=p_100.t_ext_k,
        params=p_100,
    )

    self.assertLess(
        sol_100.v_dot_hood_eff_m3_s, 0.70 * sol_200.v_dot_hood_eff_m3_s
    )

  def test_inward_makeup_air_velocity_suppresses_doorway_buoyancy_outflow(
      self,
  ) -> None:
    """High inward doorway makeup velocity floods/blocks buoyant top backflow."""
    cfg_narrow = copy.deepcopy(self.base_pb)
    cfg_narrow.geometry.room.w_door_open_mm = 120.0
    cfg_narrow.geometry.hood.d_hood_inlet_mm = 400.0
    cfg_narrow.geometry.hood.z_hood_drop_mm = 150.0
    cfg_narrow.thermal.sources.exhaust_fan_cfm = 400.0
    p_n, m_n, tf_n, qc_n, ch_n = self._make_plume_inputs(
        cfg_narrow, q_kiln_flux=600.0
    )
    sol_narrow = plume_room_lib.solve_plume_and_room(
        mesh=m_n,
        t_front_k=tf_n,
        q_conv_front_w_m2=qc_n,
        chimney=ch_n,
        t_lower_prev_k=p_n.t_ext_k,
        params=p_n,
    )

    cfg_wide = copy.deepcopy(cfg_narrow)
    cfg_wide.geometry.room.w_door_open_mm = 1800.0
    p_w, m_w, tf_w, qc_w, ch_w = self._make_plume_inputs(
        cfg_wide, q_kiln_flux=600.0
    )
    sol_wide = plume_room_lib.solve_plume_and_room(
        mesh=m_w,
        t_front_k=tf_w,
        q_conv_front_w_m2=qc_w,
        chimney=ch_w,
        t_lower_prev_k=p_w.t_ext_k,
        params=p_w,
    )

    self.assertGreater(sol_narrow.t_room_upper_k, p_n.t_ext_k + 0.5)
    self.assertLess(sol_narrow.m_dot_door_buoy_kg_s, 1e-6)
    self.assertGreater(sol_wide.m_dot_door_buoy_kg_s, 0.001)

  def test_surplus_hood_suction_extends_capture_reach_on_narrow_hood(
      self,
  ) -> None:
    """Increasing blower flow beyond V_plume increases capture even with eta_vol=1."""
    cfg_matched = copy.deepcopy(self.base_pb)
    cfg_matched.geometry.room.w_door_open_mm = 1200.0
    cfg_matched.geometry.hood.d_hood_inlet_mm = 600.0
    cfg_matched.geometry.hood.z_hood_drop_mm = 150.0
    cfg_matched.thermal.sources.exhaust_fan_cfm = 300.0
    p_m, m_m, tf_m, qc_m, ch_m = self._make_plume_inputs(
        cfg_matched, q_kiln_flux=150.0
    )
    sol_matched = plume_room_lib.solve_plume_and_room(
        mesh=m_m,
        t_front_k=tf_m,
        q_conv_front_w_m2=qc_m,
        chimney=ch_m,
        t_lower_prev_k=p_m.t_ext_k,
        params=p_m,
    )

    cfg_surplus = copy.deepcopy(cfg_matched)
    cfg_surplus.thermal.sources.exhaust_fan_cfm = 600.0
    p_s, m_s, tf_s, qc_s, ch_s = self._make_plume_inputs(
        cfg_surplus, q_kiln_flux=150.0
    )
    sol_surplus = plume_room_lib.solve_plume_and_room(
        mesh=m_s,
        t_front_k=tf_s,
        q_conv_front_w_m2=qc_s,
        chimney=ch_s,
        t_lower_prev_k=p_s.t_ext_k,
        params=p_s,
    )

    self.assertGreater(
        sol_matched.v_dot_hood_eff_m3_s, sol_matched.v_dot_plume_at_hood_m3_s
    )
    self.assertGreater(
        sol_surplus.plume_capture_fraction,
        sol_matched.plume_capture_fraction + 0.05,
    )


if __name__ == "__main__":
  absltest.main()
