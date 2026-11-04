"""Unit tests for kiln_sim.mesh."""

import math
import os
import pathlib

from absl.testing import absltest
import numpy as np
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


class MeshTest(absltest.TestCase):
  """Tests mesh discretization, surface areas, centroids, and PatchView API."""

  def setUp(self) -> None:
    super().setUp()
    config_pb, _ = io_manager.load_simulation_config(
        _find_baseline_config_path()
    )
    self.params = params_lib.from_proto(config_pb)
    self.mesh = mesh_lib.build_mesh(self.params)

  def test_surface_area_conservation(self) -> None:
    """Sum of discretized patch areas must equal exact analytical surface areas."""
    left_shield_patches = self.mesh.get_patches_by_type(
        mesh_lib.SurfaceType.SHIELD_LEFT
    )
    area_left_shield = sum(p.area_m2 for p in left_shield_patches)
    self.assertAlmostEqual(
        area_left_shield,
        self.params.w_shield_left * self.params.h_shield,
        places=6,
    )

    kiln_side_patches = self.mesh.get_patches_by_type(
        mesh_lib.SurfaceType.KILN_SIDE
    )
    area_kiln_side = sum(p.area_m2 for p in kiln_side_patches)
    expected_cyl_area = (
        math.pi * self.params.d_kiln * self.params.h_kiln_total
    )
    self.assertAlmostEqual(area_kiln_side, expected_cyl_area, places=6)

    lid_patches = self.mesh.get_patches_by_type(mesh_lib.SurfaceType.KILN_LID)
    self.assertLen(lid_patches, 1)
    self.assertAlmostEqual(
        lid_patches[0].area_m2,
        math.pi * (self.params.r_kiln ** 2),
        places=6,
    )

  def test_kiln_cylinder_centroids_lie_exactly_on_cylinder_skin(self) -> None:
    """Every KILN_SIDE centroid must sit at distance r_kiln from kiln axis."""
    kiln_patches = self.mesh.get_patches_by_type(
        mesh_lib.SurfaceType.KILN_SIDE
    )
    for p in kiln_patches:
      xc, yc, _ = p.centroid_m
      r_dist = math.hypot(
          xc - self.params.x_kiln_center,
          yc - self.params.y_kiln_center,
      )
      self.assertAlmostEqual(r_dist, self.params.r_kiln, places=6)

  def test_unit_normals_are_normalized(self) -> None:
    norms = np.linalg.norm(self.mesh.normals, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-7)

  def test_find_patch_at_coord_returns_expected_patches(self) -> None:
    # 1. Lookup left shield directly behind kiln at drum mid-height
    z_drum_mid = self.params.z_drum_bot + 0.5 * self.params.h_drum
    shield_patch = self.mesh.find_patch_at_coord(
        x_m=self.params.x_kiln_center,
        y_m=self.params.y_shield_front_left_wall,
        z_m=z_drum_mid,
        surface_type=mesh_lib.SurfaceType.SHIELD_LEFT,
    )
    self.assertIsNotNone(shield_patch)
    assert shield_patch is not None
    self.assertEqual(
        shield_patch.surface_type, mesh_lib.SurfaceType.SHIELD_LEFT
    )
    self.assertEqual(shield_patch.normal, (0.0, 1.0, 0.0))

    # 2. Lookup kiln front face (+x direction) at drum mid-height
    kiln_front_patch = self.mesh.find_patch_at_coord(
        x_m=self.params.x_kiln_center + self.params.r_kiln,
        y_m=self.params.y_kiln_center,
        z_m=z_drum_mid,
        surface_type=mesh_lib.SurfaceType.KILN_SIDE,
        tol_m=0.05,
    )
    self.assertIsNotNone(kiln_front_patch)
    assert kiln_front_patch is not None
    self.assertEqual(
        kiln_front_patch.surface_type, mesh_lib.SurfaceType.KILN_SIDE
    )
    self.assertGreater(kiln_front_patch.normal[0], 0.95)

    # 3. Lookup in empty middle of room returns None
    empty_lookup = self.mesh.find_patch_at_coord(x_m=2.0, y_m=2.0, z_m=1.5)
    self.assertIsNone(empty_lookup)


if __name__ == "__main__":
  absltest.main()