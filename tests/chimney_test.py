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


if __name__ == "__main__":
  absltest.main()