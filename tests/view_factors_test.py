"""Unit tests for kiln_sim.view_factors."""

import copy
import os
import pathlib

from absl.testing import absltest
import numpy as np
from src import io_manager
from src import mesh as mesh_lib
from src import params as params_lib
from src import view_factors as vf_lib


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


class ViewFactorsTest(absltest.TestCase):
  """Tests view factor analytical limits, conservation, and butt-joint geometry."""

  def setUp(self) -> None:
    super().setUp()
    self.base_pb, _ = io_manager.load_simulation_config(
        _find_baseline_config_path()
    )
    self.base_pb.grid.dx_wall_mm = 100.0
    self.base_pb.grid.dz_wall_mm = 100.0
    self.params = params_lib.from_proto(self.base_pb)
    self.mesh = mesh_lib.build_mesh(self.params)
    self.vf_sys = vf_lib.compute_view_factors(self.mesh, self.params)

  def test_coaxial_parallel_disks_analytical_limits(self) -> None:
    """Two equal disks have F -> 1 as separation h -> 0 and F -> 0 as h -> inf."""
    vf_touching = vf_lib.coaxial_parallel_disks_vf(r1=0.35, r2=0.35, h=1e-6)
    self.assertAlmostEqual(vf_touching, 1.0, places=4)

    vf_far = vf_lib.coaxial_parallel_disks_vf(r1=0.35, r2=0.35, h=100.0)
    self.assertLess(vf_far, 1e-4)

    vf_baseline = vf_lib.coaxial_parallel_disks_vf(
        r1=0.35, r2=0.365, h=0.210
    )
    self.assertBetween(vf_baseline, 0.55, 0.65)

  def test_non_negativity_and_row_sum_conservation(self) -> None:
    """Every patch row must sum to 1.0 when including f_room."""
    self.assertTrue(np.all(self.vf_sys.f_eff >= 0.0))
    self.assertTrue(np.all(self.vf_sys.f_room >= 0.0))
    row_totals = np.sum(self.vf_sys.f_eff, axis=1) + self.vf_sys.f_room
    np.testing.assert_allclose(row_totals, 1.0, atol=1e-12)

  def test_corner_butt_joint_view_factors_between_left_and_back_shields(
      self,
  ) -> None:
    """Left shield starts at x=d_chimney+t_shield (45 mm) vs back at y=d_chimney (25 mm)."""
    stypes = self.mesh.surface_types
    kiln_mask = stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
    left_mask = stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT)
    back_mask = stypes == int(mesh_lib.SurfaceType.SHIELD_BACK)

    af_kiln_to_left = float(
        np.sum(
            self.mesh.areas[kiln_mask, None]
            * self.vf_sys.f_eff[np.ix_(kiln_mask, left_mask)]
        )
    )
    af_kiln_to_back = float(
        np.sum(
            self.mesh.areas[kiln_mask, None]
            * self.vf_sys.f_eff[np.ix_(kiln_mask, back_mask)]
        )
    )
    # Because SHIELD_LEFT spans [45 mm, 845 mm] (extending 20 mm further past
    # the kiln centerline x=600 mm than SHIELD_BACK's [25 mm, 825 mm]),
    # SHIELD_LEFT intercepts slightly more kiln radiation (~0.0031 m^2 difference).
    self.assertGreater(af_kiln_to_left, af_kiln_to_back)
    self.assertAlmostEqual(af_kiln_to_left, af_kiln_to_back, delta=0.01)

  def test_polished_reflector_increases_kiln_bottom_to_shield_exchange(self) -> None:
    """High-reflectivity stainless reflector bounces kiln base rays to shields."""
    cfg_black = copy.deepcopy(self.base_pb)
    cfg_black.thermal.emissivities.eps_reflector_top = 0.99  # rho = 0.01
    p_black = params_lib.from_proto(cfg_black)
    vf_black = vf_lib.compute_view_factors(self.mesh, p_black)

    stypes = self.mesh.surface_types
    bot_id = int(
        np.flatnonzero(stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))[0]
    )
    shield_mask = (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT)) | (
        stypes == int(mesh_lib.SurfaceType.SHIELD_BACK)
    )

    vf_bot_to_shields_polished = float(
        np.sum(self.vf_sys.f_eff[bot_id, shield_mask])
    )
    vf_bot_to_shields_black = float(
        np.sum(vf_black.f_eff[bot_id, shield_mask])
    )
    self.assertGreater(
        vf_bot_to_shields_polished, vf_bot_to_shields_black * 5.0
    )

  def test_elevated_reflector_occludes_kiln_bottom_from_vinyl_edge_sensor(
      self,
  ) -> None:
    """Raising a reflector closer to the kiln bottom widens its shadow onto vinyl_edge."""
    # With a 600 mm reflector (r = 300 mm < r_kiln = 350 mm) at low elevation
    # (g_floor_reflector = 20 mm), the overhanging outer ring of the 700 mm kiln
    # bottom (r in [300 mm, 350 mm]) has an unobstructed line of sight past the
    # reflector rim to vinyl_edge_x (at x = 1005 mm, r = 405 mm from kiln axis).
    # When the 600 mm reflector is raised close to the kiln bottom
    # (g_floor_reflector = 190 mm, just below h_kiln_base = 210 mm), the shadow
    # cone of the reflector rim widens and blocks more of the kiln bottom from
    # reaching vinyl_edge_x.
    cfg_low = copy.deepcopy(self.base_pb)
    cfg_low.geometry.floor.d_reflector_mm = 600.0
    cfg_low.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_low.geometry.floor.g_floor_reflector_mm = 20.0
    p_low = params_lib.from_proto(cfg_low)
    m_low = mesh_lib.build_mesh(p_low)
    vf_low = vf_lib.compute_view_factors(m_low, p_low)

    cfg_high = copy.deepcopy(cfg_low)
    cfg_high.geometry.floor.g_floor_reflector_mm = 190.0
    p_high = params_lib.from_proto(cfg_high)
    m_high = mesh_lib.build_mesh(p_high)
    vf_high = vf_lib.compute_view_factors(m_high, p_high)

    bot_id = int(
        np.flatnonzero(
            m_low.surface_types == int(mesh_lib.SurfaceType.KILN_BOTTOM)
        )[0]
    )
    vf_vinyl_to_bot_low = float(vf_low.sensor_vf["vinyl_edge_x"][bot_id])
    vf_vinyl_to_bot_high = float(vf_high.sensor_vf["vinyl_edge_x"][bot_id])

    self.assertGreater(vf_vinyl_to_bot_low, 0.0)
    self.assertLess(vf_vinyl_to_bot_high, vf_vinyl_to_bot_low)

  def test_wider_reflector_occludes_kiln_bottom_at_pad_edge(
      self,
  ) -> None:
    """Increasing reflector diameter past the kiln diameter blocks KILN_BOTTOM at pad_edge."""
    # Because the reflector (z = z_reflector_top) sits between the floor pad
    # (z = z_pad_top) and the kiln bottom (z = z_drum_bot), any ray from the
    # kiln bottom (r <= r_kiln = 350 mm) to pad_edge (at r = r_reflector + 5 mm,
    # z = z_pad_top) must pass through the plane z = z_reflector_top at a radial
    # coordinate strictly between r_kiln and r_reflector + 5 mm. Whenever
    # r_reflector >= r_kiln + 5 mm (e.g., d_reflector = 1000 mm, r = 500 mm),
    # the opaque reflector disk completely blocks direct radiation from the kiln
    # bottom to pad_edge.
    cfg_wide = copy.deepcopy(self.base_pb)
    cfg_wide.geometry.floor.d_reflector_mm = 1000.0
    cfg_wide.geometry.floor.w_floor_pad_mm = 1200.0
    cfg_wide.geometry.floor.g_floor_reflector_mm = 20.0
    cfg_wide.geometry.floor.t_reflector_calsil_mm = 20.0
    p_wide = params_lib.from_proto(cfg_wide)
    m_wide = mesh_lib.build_mesh(p_wide)
    vf_wide = vf_lib.compute_view_factors(m_wide, p_wide)

    bot_id = int(
        np.flatnonzero(
            m_wide.surface_types == int(mesh_lib.SurfaceType.KILN_BOTTOM)
        )[0]
    )
    self.assertAlmostEqual(
        float(vf_wide.sensor_vf["pad_edge_room"][bot_id]), 0.0, places=6
    )
    self.assertAlmostEqual(
        float(vf_wide.sensor_vf["pad_edge_corner"][bot_id]), 0.0, places=6
    )

  def test_pad_edge_view_of_kiln_bottom_depends_on_reflector_height(
      self,
  ) -> None:
    """Changing reflector height between pad and kiln bottom changes pad_edge view factor."""
    # When a 600 mm reflector (r = 300 mm < r_kiln = 350 mm) is near the pad
    # (g_floor_reflector = 20 mm), the outer rim of the 700 mm kiln bottom
    # (r in [300 mm, 350 mm]) overhangs the reflector and shines directly onto
    # pad_edge_room (at r = 305 mm). Raising the reflector closer to the kiln
    # bottom (g_floor_reflector = 180 mm) changes the shadow angle and reduces
    # the visible portion of the kiln bottom from pad_edge_room.
    cfg_low = copy.deepcopy(self.base_pb)
    cfg_low.geometry.floor.d_reflector_mm = 600.0
    cfg_low.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_low.geometry.floor.g_floor_reflector_mm = 20.0
    p_low = params_lib.from_proto(cfg_low)
    m_low = mesh_lib.build_mesh(p_low)
    vf_low = vf_lib.compute_view_factors(m_low, p_low)

    cfg_high = copy.deepcopy(cfg_low)
    cfg_high.geometry.floor.g_floor_reflector_mm = 180.0
    p_high = params_lib.from_proto(cfg_high)
    m_high = mesh_lib.build_mesh(p_high)
    vf_high = vf_lib.compute_view_factors(m_high, p_high)

    bot_id = int(
        np.flatnonzero(
            m_low.surface_types == int(mesh_lib.SurfaceType.KILN_BOTTOM)
        )[0]
    )
    vf_pad_low = float(vf_low.sensor_vf["pad_edge_room"][bot_id])
    vf_pad_high = float(vf_high.sensor_vf["pad_edge_room"][bot_id])

    self.assertGreater(vf_pad_low, vf_pad_high)

  def test_wide_reflector_couples_radiatively_to_ssr_box(self) -> None:
    """When r_reflector > r_kiln, the overhanging reflector annulus sees SSR_BOX."""
    cfg_wide = copy.deepcopy(self.base_pb)
    cfg_wide.geometry.floor.d_reflector_mm = 1000.0
    cfg_wide.geometry.floor.w_floor_pad_mm = 1200.0
    p_wide = params_lib.from_proto(cfg_wide)
    m_wide = mesh_lib.build_mesh(p_wide)
    vf_wide = vf_lib.compute_view_factors(m_wide, p_wide)

    stypes = m_wide.surface_types
    ssr_mask = stypes == int(mesh_lib.SurfaceType.SSR_BOX)
    ref_mask = stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
    kiln_mask = (stypes == int(mesh_lib.SurfaceType.KILN_SIDE)) | (
        stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM)
    )

    # 1. Diffuse view factor between SSR_BOX and the overhanging REFLECTOR_TOP
    # annulus must be strictly positive.
    af_ssr_to_ref = float(
        np.sum(
            m_wide.areas[ssr_mask, None]
            * vf_wide.f_eff[np.ix_(ssr_mask, ref_mask)]
        )
    )
    self.assertGreater(af_ssr_to_ref, 0.0)

    # 2. Specular reflection off the polished overhanging reflector annulus
    # must increase effective radiative exchange between the kiln (KILN_SIDE +
    # KILN_BOTTOM) and SSR_BOX compared to a matte/black reflector.
    cfg_black = copy.deepcopy(cfg_wide)
    cfg_black.thermal.emissivities.eps_reflector_top = 0.99  # rho = 0.01
    p_black = params_lib.from_proto(cfg_black)
    vf_black = vf_lib.compute_view_factors(m_wide, p_black)

    af_kiln_to_ssr_polished = float(
        np.sum(
            m_wide.areas[kiln_mask, None]
            * vf_wide.f_eff[np.ix_(kiln_mask, ssr_mask)]
        )
    )
    af_kiln_to_ssr_black = float(
        np.sum(
            m_wide.areas[kiln_mask, None]
            * vf_black.f_eff[np.ix_(kiln_mask, ssr_mask)]
        )
    )
    self.assertGreater(af_kiln_to_ssr_polished, af_kiln_to_ssr_black)

  def test_wide_polished_reflector_increases_kiln_to_wall_and_gypsum_exchange(
      self,
  ) -> None:
    """Overhanging polished reflector bounces kiln side & bottom rays to walls."""
    cfg_narrow = copy.deepcopy(self.base_pb)
    cfg_narrow.geometry.floor.d_reflector_mm = 700.0  # r_ref == r_kiln (350 mm)
    cfg_narrow.geometry.floor.w_floor_pad_mm = 1200.0
    p_narrow = params_lib.from_proto(cfg_narrow)
    m_narrow = mesh_lib.build_mesh(p_narrow)
    vf_narrow = vf_lib.compute_view_factors(m_narrow, p_narrow)

    cfg_wide = copy.deepcopy(cfg_narrow)
    cfg_wide.geometry.floor.d_reflector_mm = 1000.0  # r_ref = 500 mm
    p_wide = params_lib.from_proto(cfg_wide)
    m_wide = mesh_lib.build_mesh(p_wide)
    vf_wide = vf_lib.compute_view_factors(m_wide, p_wide)

    stypes = m_wide.surface_types
    kiln_side_mask = stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
    kiln_mask = kiln_side_mask | (
        stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM)
    )
    shield_mask = (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT)) | (
        stypes == int(mesh_lib.SurfaceType.SHIELD_BACK)
    )
    gypsum_mask = (stypes == int(mesh_lib.SurfaceType.GYPSUM_LEFT)) | (
        stypes == int(mesh_lib.SurfaceType.GYPSUM_BACK)
    )

    af_kiln_to_shield_narrow = float(
        np.sum(
            m_narrow.areas[kiln_mask, None]
            * vf_narrow.f_eff[np.ix_(kiln_mask, shield_mask)]
        )
    )
    af_kiln_to_shield_wide = float(
        np.sum(
            m_wide.areas[kiln_mask, None]
            * vf_wide.f_eff[np.ix_(kiln_mask, shield_mask)]
        )
    )
    self.assertGreater(af_kiln_to_shield_wide, af_kiln_to_shield_narrow)

    af_kiln_to_gypsum_narrow = float(
        np.sum(
            m_narrow.areas[kiln_mask, None]
            * vf_narrow.f_eff[np.ix_(kiln_mask, gypsum_mask)]
        )
    )
    af_kiln_to_gypsum_wide = float(
        np.sum(
            m_wide.areas[kiln_mask, None]
            * vf_wide.f_eff[np.ix_(kiln_mask, gypsum_mask)]
        )
    )
    self.assertGreater(af_kiln_to_gypsum_wide, af_kiln_to_gypsum_narrow)

    # Specifically verify that KILN_SIDE reflects off the overhanging annulus
    # onto both the shield walls and the gypsum walls when polished vs black.
    cfg_wide_black = copy.deepcopy(cfg_wide)
    cfg_wide_black.thermal.emissivities.eps_reflector_top = 0.99
    p_wide_black = params_lib.from_proto(cfg_wide_black)
    vf_wide_black = vf_lib.compute_view_factors(m_wide, p_wide_black)

    af_ks_to_shield_polished = float(
        np.sum(
            m_wide.areas[kiln_side_mask, None]
            * vf_wide.f_eff[np.ix_(kiln_side_mask, shield_mask)]
        )
    )
    af_ks_to_shield_black = float(
        np.sum(
            m_wide.areas[kiln_side_mask, None]
            * vf_wide_black.f_eff[np.ix_(kiln_side_mask, shield_mask)]
        )
    )
    self.assertGreater(af_ks_to_shield_polished, af_ks_to_shield_black)

    af_ks_to_gypsum_polished = float(
        np.sum(
            m_wide.areas[kiln_side_mask, None]
            * vf_wide.f_eff[np.ix_(kiln_side_mask, gypsum_mask)]
        )
    )
    af_ks_to_gypsum_black = float(
        np.sum(
            m_wide.areas[kiln_side_mask, None]
            * vf_wide_black.f_eff[np.ix_(kiln_side_mask, gypsum_mask)]
        )
    )
    self.assertGreater(af_ks_to_gypsum_polished, af_ks_to_gypsum_black)

  def test_wide_polished_reflector_increases_ceiling_sensor_view_of_kiln(
      self,
  ) -> None:
    """Overhanging polished reflector bounces kiln radiation to ceiling sensors."""
    cfg_narrow = copy.deepcopy(self.base_pb)
    cfg_narrow.geometry.floor.d_reflector_mm = 700.0  # r_ref == r_kiln (350 mm)
    cfg_narrow.geometry.floor.w_floor_pad_mm = 1200.0
    p_narrow = params_lib.from_proto(cfg_narrow)
    m_narrow = mesh_lib.build_mesh(p_narrow)
    vf_narrow = vf_lib.compute_view_factors(m_narrow, p_narrow)

    cfg_wide = copy.deepcopy(cfg_narrow)
    cfg_wide.geometry.floor.d_reflector_mm = 1000.0  # r_ref = 500 mm
    p_wide = params_lib.from_proto(cfg_wide)
    m_wide = mesh_lib.build_mesh(p_wide)
    vf_wide = vf_lib.compute_view_factors(m_wide, p_wide)

    stypes = m_wide.surface_types
    ref_id = int(
        np.flatnonzero(stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP))[0]
    )
    kiln_mask = (stypes == int(mesh_lib.SurfaceType.KILN_SIDE)) | (
        stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM)
    )

    # 1. When r_reflector == r_kiln (700 mm), the reflector sits entirely
    # underneath the solid kiln drum and cannot be seen from the ceiling.
    for sensor_name in (
        "ceiling_hood_rim_corner",
        "ceiling_hood_rim_room",
        "ceiling_chimney_corner",
    ):
      self.assertAlmostEqual(
          float(vf_narrow.sensor_vf[sensor_name][ref_id]), 0.0, places=6
      )

    # 2. When r_reflector > r_kiln (1000 mm), the overhanging polished annulus
    # reflects kiln radiation up to the ceiling sensors, increasing their view
    # factor to the kiln compared to the narrow reflector.
    vf_ceil_kiln_narrow = float(
        np.sum(vf_narrow.sensor_vf["ceiling_chimney_corner"][kiln_mask])
    )
    vf_ceil_kiln_wide = float(
        np.sum(vf_wide.sensor_vf["ceiling_chimney_corner"][kiln_mask])
    )
    self.assertGreater(vf_ceil_kiln_wide, vf_ceil_kiln_narrow)

  def test_zero_reflector_diameter_conserves_view_factors_without_nans(
      self,
  ) -> None:
    """Setting d_reflector_mm = 0 produces finite, conserved view factors."""
    cfg_no_ref = copy.deepcopy(self.base_pb)
    cfg_no_ref.geometry.floor.d_reflector_mm = 0.0
    cfg_no_ref.geometry.floor.t_reflector_calsil_mm = 0.0
    cfg_no_ref.geometry.floor.g_floor_reflector_mm = 0.0
    p_no_ref = params_lib.from_proto(cfg_no_ref)
    m_no_ref = mesh_lib.build_mesh(p_no_ref)
    vf_no_ref = vf_lib.compute_view_factors(m_no_ref, p_no_ref)

    self.assertFalse(np.any(np.isnan(vf_no_ref.f_eff)))
    self.assertFalse(np.any(np.isnan(vf_no_ref.f_room)))
    self.assertTrue(np.all(vf_no_ref.f_eff >= 0.0))
    row_totals = np.sum(vf_no_ref.f_eff, axis=1) + vf_no_ref.f_room
    np.testing.assert_allclose(row_totals, 1.0, atol=1e-12)

    bot_id = int(
        np.flatnonzero(
            m_no_ref.surface_types == int(mesh_lib.SurfaceType.KILN_BOTTOM)
        )[0]
    )
    # Without a reflector, the pad center directly under the kiln has an
    # unobstructed view of KILN_BOTTOM.
    self.assertGreater(float(vf_no_ref.sensor_vf["pad_center"][bot_id]), 0.5)


if __name__ == "__main__":
  absltest.main()