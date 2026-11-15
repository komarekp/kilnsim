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


if __name__ == "__main__":
  absltest.main()
