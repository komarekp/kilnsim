"""Unit tests for kiln_sim.chimney."""

import copy
import os
import pathlib

from absl.testing import absltest
import numpy as np
from src import chimney as chimney_lib
from src import io_manager
from src import mesh as mesh_lib
from src import params as params_lib


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


class ChimneyTest(absltest.TestCase):
  """Tests 1D buoyant channel draft and heat transfer behind the shields."""

  def setUp(self) -> None:
    super().setUp()
    self.base_pb, _ = io_manager.load_simulation_config(
        _find_baseline_config_path()
    )
    self.params = params_lib.from_proto(self.base_pb)
    self.mesh = mesh_lib.build_mesh(self.params)

  def test_cold_isothermal_shield_produces_zero_heating(self) -> None:
    """When shield is at ambient 20 C, exit air and AAC wall stay at 20 C."""
    t_amb_k = self.params.t_ext_k  # 293.15 K (20.0 C)
    t_shield_back = np.full(self.mesh.num_patches, t_amb_k, dtype=np.float64)

    sol = chimney_lib.solve_chimney(
        t_shield_back_k=t_shield_back,
        t_inlet_k=t_amb_k,
        mesh=self.mesh,
        params=self.params,
    )

    np.testing.assert_allclose(sol.t_exit_left_k, t_amb_k, atol=1e-5)
    np.testing.assert_allclose(sol.t_exit_back_k, t_amb_k, atol=1e-5)
    np.testing.assert_allclose(sol.t_aac_front_patch_k, t_amb_k, atol=1e-5)
    self.assertAlmostEqual(sol.q_conv_total_w, 0.0, places=5)

  def test_hotter_shield_monotonically_increases_draft_and_exit_temp(self) -> None:
    """Increasing shield back temperature from 45 C to 95 C increases draft."""
    t_amb_k = self.params.t_ext_k
    t_warm = np.full(
        self.mesh.num_patches, t_amb_k + 25.0, dtype=np.float64
    )
    t_hot = np.full(
        self.mesh.num_patches, t_amb_k + 75.0, dtype=np.float64
    )

    sol_warm = chimney_lib.solve_chimney(
        t_shield_back_k=t_warm,
        t_inlet_k=t_amb_k,
        mesh=self.mesh,
        params=self.params,
    )
    sol_hot = chimney_lib.solve_chimney(
        t_shield_back_k=t_hot,
        t_inlet_k=t_amb_k,
        mesh=self.mesh,
        params=self.params,
    )

    self.assertGreater(
        float(np.mean(sol_hot.v_gap_left)),
        float(np.mean(sol_warm.v_gap_left)),
    )
    self.assertGreater(
        float(np.mean(sol_hot.t_exit_left_k)),
        float(np.mean(sol_warm.t_exit_left_k)),
    )
    self.assertGreater(sol_hot.q_conv_total_w, sol_warm.q_conv_total_w)

  def test_unsealed_chimney_sides_dilute_outermost_column(self) -> None:
    """When seal_chimney_sides is False, lateral inflow cools the outer column."""
    t_amb_k = self.params.t_ext_k
    t_hot = np.full(
        self.mesh.num_patches, t_amb_k + 50.0, dtype=np.float64
    )

    cfg_unsealed = copy.deepcopy(self.base_pb)
    cfg_unsealed.geometry.room.chimney.seal_chimney_sides = False
    params_unsealed = params_lib.from_proto(cfg_unsealed)

    sol_sealed = chimney_lib.solve_chimney(
        t_shield_back_k=t_hot,
        t_inlet_k=t_amb_k,
        mesh=self.mesh,
        params=self.params,
    )
    sol_unsealed = chimney_lib.solve_chimney(
        t_shield_back_k=t_hot,
        t_inlet_k=t_amb_k,
        mesh=self.mesh,
        params=params_unsealed,
    )

    # Outer column is index -1
    self.assertLess(
        sol_unsealed.t_exit_left_k[-1],
        sol_sealed.t_exit_left_k[-1],
    )

  def test_sealed_stagnant_cavity_matches_1d_air_conduction_and_radiation(
      self,
  ) -> None:
    """Sealed chimney in low-Ra conduction limit transfers k_air/d_eff + rad."""
    cfg_sealed = copy.deepcopy(self.base_pb)
    cfg_sealed.geometry.shield.g_floor_mm = 0.0
    cfg_sealed.geometry.shield.g_ceil_mm = 0.0
    cfg_sealed.geometry.room.chimney.d_chimney_mm = 50.0
    cfg_sealed.geometry.room.chimney.seal_chimney_sides = True
    params_sealed = params_lib.from_proto(cfg_sealed)

    dt_shield = 0.2  # K -> Ra_gap << critical Rayleigh threshold
    t_shield_k = np.full(
        self.mesh.num_patches,
        params_sealed.t_ext_k + dt_shield,
        dtype=np.float64,
    )
    sol = chimney_lib.solve_chimney(
        t_shield_back_k=t_shield_k,
        t_inlet_k=params_sealed.t_ext_k,
        mesh=self.mesh,
        params=params_sealed,
    )

    shield_mask = self.mesh.chimney_wall_id >= 0
    np.testing.assert_allclose(sol.v_gap_left, 0.0, atol=1e-9)
    np.testing.assert_allclose(sol.v_gap_back, 0.0, atol=1e-9)

    # Pure air conduction gap conductance across d_eff is 0.5 * h_chan = k_air / d_eff.
    expected_gap_u_conv = chimney_lib.AIR_K / params_sealed.d_chimney_eff
    actual_gap_u_conv = 0.5 * float(np.mean(sol.h_chan_patch[shield_mask]))
    self.assertAlmostEqual(actual_gap_u_conv, expected_gap_u_conv, delta=0.05)

    # Total heat flux across sealed cavity + AAC wall must match analytical series.
    eps_eff = 1.0 / (
        (1.0 / params_sealed.eps_shield) + (1.0 / params_sealed.eps_aac) - 1.0
    )
    t_mean_k = params_sealed.t_ext_k + 0.5 * dt_shield
    h_rad = 4.0 * eps_eff * params_lib.STEFAN_BOLTZMANN * (t_mean_k**3)
    u_gap_total = expected_gap_u_conv + h_rad
    u_series = 1.0 / ((1.0 / u_gap_total) + (1.0 / params_sealed.u_aac_wall))
    expected_q = u_series * dt_shield
    actual_q = float(np.mean(sol.q_back_from_shield_w_m2[shield_mask]))
    self.assertAlmostEqual(actual_q, expected_q, delta=0.02)

  def test_wide_open_chimney_is_not_clamped_by_arbitrary_h_floor(self) -> None:
    """Wide open channels with low Nusselt-predicted h avoid a 3.5 W/m^2K clamp."""
    cfg_wide = copy.deepcopy(self.base_pb)
    cfg_wide.geometry.room.chimney.d_chimney_mm = 80.0
    cfg_wide.geometry.floor.w_floor_pad_mm = 1200.0
    params_wide = params_lib.from_proto(cfg_wide)

    t_shield_k = np.full(
        self.mesh.num_patches, params_wide.t_ext_k + 1.0, dtype=np.float64
    )
    sol = chimney_lib.solve_chimney(
        t_shield_back_k=t_shield_k,
        t_inlet_k=params_wide.t_ext_k,
        mesh=self.mesh,
        params=params_wide,
    )
    shield_mask = self.mesh.chimney_wall_id >= 0
    self.assertLess(float(np.mean(sol.h_chan_patch[shield_mask])), 3.0)


if __name__ == "__main__":
  absltest.main()