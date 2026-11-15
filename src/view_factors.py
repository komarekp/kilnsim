"""Radiative view factor matrix and specular floor-reflector engine.

Computes the diffuse patch-to-patch view factor matrix F_direct, the specular
virtual-image matrix F_mirror reflected off the circular polished stainless
steel floor reflector, and differential point-to-patch view factors for the
floor and ceiling safety sensors.
"""

import dataclasses
import math
import numpy as np
from src import constants
from src import mesh as mesh_lib
from src import params as params_lib

# Spatial offsets for ceiling point sensors at z = h_wall.
CEILING_SENSOR_RIM_OFFSET_M: float = constants.CEILING_SENSOR_RIM_OFFSET_M
CEILING_SENSOR_WALL_OFFSET_M: float = constants.CEILING_SENSOR_WALL_OFFSET_M

# Quadrature grid resolution for integrating view factors across finite disks.
_KILN_DISK_QUAD_R: int = constants.KILN_DISK_QUAD_R
_KILN_DISK_QUAD_THETA: int = constants.KILN_DISK_QUAD_THETA


@dataclasses.dataclass(frozen=True)
class ViewFactorSystem:
  """Container for precomputed radiative view factor matrices and vectors.

  Attributes:
    f_eff: (N, N) effective view factor matrix combining direct diffuse exchange
      and specular reflection from the circular floor reflector.
    f_room: (N,) residual view factor from each patch to the open room ambient
      enclosure, satisfying row conservation `sum_n(f_eff[m, n]) + f_room[m] = 1`.
    sensor_vf: Mapping from floor or ceiling sensor name to its (N,)
      differential view factor vector `dF_{sensor -> n}` across all mesh patches.
  """

  f_eff: np.ndarray
  f_room: np.ndarray
  sensor_vf: dict[str, np.ndarray]


def coaxial_parallel_disks_vf(r1: float, r2: float, h: float) -> float:
  """Computes the exact analytical view factor between coaxial parallel disks.

  Args:
    r1: Radius of emitting disk 1 in metres.
    r2: Radius of receiving disk 2 in metres.
    h: Perpendicular axial separation distance in metres.

  Returns:
    Dimensionless view factor F_{1 -> 2} in [0.0, 1.0].
  """
  if r1 <= 0.0 or r2 <= 0.0 or h <= 0.0:
    return 0.0
  cap_r1 = r1 / h
  cap_r2 = r2 / h
  s = 1.0 + (1.0 + cap_r2 * cap_r2) / (cap_r1 * cap_r1)
  discriminant = max(0.0, s * s - 4.0 * (r2 / r1) ** 2)
  vf = 0.5 * (s - math.sqrt(discriminant))
  return max(0.0, min(1.0, vf))


def _is_segment_blocked_by_reflector_disk(
    p1: np.ndarray,
    p2: np.ndarray,
    params: params_lib.SimParams,
) -> np.ndarray:
  """Returns boolean mask of line segments p1 -> p2 occluded by the reflector.

  The circular reflector assembly spans elevations [z_reflector_bot,
  z_reflector_top] with radius r_reflector centered at (x_kiln_center,
  y_kiln_center). A ray starting at or below z_reflector_top on the floor and
  ending above z_reflector_top is blocked if it either originates directly
  underneath a flush reflector (z_reflector_bot <= p1_z and r_p1 <= r_reflector)
  or intersects the horizontal plane z = z_reflector_top (or z = z_reflector_bot)
  within radius r_reflector.

  Args:
    p1: (3,) or (M, 3) array of ray origin coordinates in metres.
    p2: (M, 3) array of ray target coordinates in metres.
    params: Validated SimParams instance.

  Returns:
    (M,) boolean array where True indicates the ray is blocked by the reflector.
  """
  if params.r_reflector <= 0.0:
    return np.zeros(p2.shape[0], dtype=bool)

  p1_arr = np.broadcast_to(p1, p2.shape)
  dz = p2[:, 2] - p1_arr[:, 2]
  blocked = np.zeros(p2.shape[0], dtype=bool)

  for z_plane in (params.z_reflector_bot, params.z_reflector_top):
    # Check rays whose endpoints straddle z_plane (in either direction).
    z_low = np.minimum(p1_arr[:, 2], p2[:, 2])
    z_high = np.maximum(p1_arr[:, 2], p2[:, 2])
    crosses = (
        (z_low < z_plane - 1e-7)
        & (z_high > z_plane + 1e-7)
        & (np.abs(dz) > 1e-9)
    )
    if not np.any(crosses):
      continue
    t = np.clip((z_plane - p1_arr[crosses, 2]) / dz[crosses], 0.0, 1.0)
    x_hit = p1_arr[crosses, 0] + t * (p2[crosses, 0] - p1_arr[crosses, 0])
    y_hit = p1_arr[crosses, 1] + t * (p2[crosses, 1] - p1_arr[crosses, 1])
    r_hit = np.hypot(
        x_hit - params.x_kiln_center,
        y_hit - params.y_kiln_center,
    )
    blocked_cross = r_hit <= params.r_reflector + 1e-7
    blocked_indices = np.flatnonzero(crosses)
    blocked[blocked_indices] |= blocked_cross

  # Also block any ray that passes through the cylindrical side rim of the
  # reflector assembly (r = r_reflector, z in [z_reflector_bot, z_reflector_top])
  # when the reflector has finite thickness or is seated flush on the pad.
  has_flush_zero_thick_contact = (
      params.g_floor_reflector <= 1e-7 and params.t_reflector_calsil <= 1e-7
  )
  z_check_bot = (
      params.z_pad_top
      if params.g_floor_reflector <= 1e-7
      else params.z_reflector_bot
  )
  z_check_top = params.z_reflector_top

  # Solve 2D radial intersection with circle of radius r_reflector around
  # (x_kiln_center, y_kiln_center) in the xy-plane along segment t in [0, 1].
  ox = p1_arr[:, 0] - params.x_kiln_center
  oy = p1_arr[:, 1] - params.y_kiln_center
  dx = p2[:, 0] - p1_arr[:, 0]
  dy = p2[:, 1] - p1_arr[:, 1]

  a_quad = dx * dx + dy * dy
  b_quad = 2.0 * (ox * dx + oy * dy)
  c_quad = ox * ox + oy * oy - (params.r_reflector ** 2)
  disc = b_quad * b_quad - 4.0 * a_quad * c_quad

  valid_cyl = (
      (not has_flush_zero_thick_contact)
      & (z_check_top > z_check_bot + 1e-7)
      & (a_quad > 1e-12)
      & (disc >= 0.0)
  )
  if np.any(valid_cyl):
    sqrt_disc = np.sqrt(np.maximum(0.0, disc[valid_cyl]))
    a_v = a_quad[valid_cyl]
    b_v = b_quad[valid_cyl]
    t1 = (-b_v - sqrt_disc) / (2.0 * a_v)
    t2 = (-b_v + sqrt_disc) / (2.0 * a_v)
    # Segment intervals inside cylinder r <= r_reflector are [max(0, t1), min(1, t2)]
    t_enter = np.maximum(0.0, np.minimum(t1, t2))
    t_exit = np.minimum(1.0, np.maximum(t1, t2))
    has_overlap = t_exit >= t_enter

    z1_v = p1_arr[valid_cyl, 2]
    dz_v = dz[valid_cyl]
    z_at_enter = z1_v + t_enter * dz_v
    z_at_exit = z1_v + t_exit * dz_v
    z_seg_min = np.minimum(z_at_enter, z_at_exit)
    z_seg_max = np.maximum(z_at_enter, z_at_exit)

    # Check if the z-interval inside r <= r_reflector overlaps [z_check_bot, z_check_top]
    z_overlaps = (z_seg_max >= z_check_bot - 1e-7) & (
        z_seg_min <= z_check_top + 1e-7
    )
    cyl_indices = np.flatnonzero(valid_cyl)
    blocked[cyl_indices] |= has_overlap & z_overlaps

  return blocked


def _is_segment_blocked_by_kiln_cylinder(
    p1: np.ndarray,
    p2: np.ndarray,
    params: params_lib.SimParams,
) -> np.ndarray:
  """Returns boolean mask of line segments p1 -> p2 occluded by the kiln drum.

  The solid cylindrical kiln body spans radial coordinates r <= r_kiln around
  (x_kiln_center, y_kiln_center) and elevations z in [z_drum_bot, z_kiln_top].
  A segment p(t) = p1 + t * (p2 - p1), t in (0, 1), is blocked if there is a
  non-empty sub-interval of t in (0, 1) where both z(t) in [z_drum_bot,
  z_kiln_top] and r_xy(t) < r_kiln.

  Args:
    p1: (3,) or (M, 3) array of ray origin coordinates in metres.
    p2: (M, 3) array of ray target coordinates in metres.
    params: Validated SimParams instance.

  Returns:
    (M,) boolean array where True indicates the ray intersects the kiln drum.
  """
  r_cyl = params.r_kiln - 1e-4
  z_bot = params.z_drum_bot + 1e-5
  z_top = params.z_kiln_top - 1e-5
  if r_cyl <= 0.0 or z_top <= z_bot:
    return np.zeros(p2.shape[0], dtype=bool)

  p1_arr = np.broadcast_to(p1, p2.shape)
  n_seg = p2.shape[0]
  dz = p2[:, 2] - p1_arr[:, 2]

  t_z_enter = np.ones(n_seg, dtype=np.float64)
  t_z_exit = np.zeros(n_seg, dtype=np.float64)

  non_horiz = np.abs(dz) > 1e-9
  if np.any(non_horiz):
    inv_dz = 1.0 / dz[non_horiz]
    tz1 = (z_bot - p1_arr[non_horiz, 2]) * inv_dz
    tz2 = (z_top - p1_arr[non_horiz, 2]) * inv_dz
    t_z_enter[non_horiz] = np.clip(np.minimum(tz1, tz2), 0.0, 1.0)
    t_z_exit[non_horiz] = np.clip(np.maximum(tz1, tz2), 0.0, 1.0)

  horiz_in_z = (~non_horiz) & (p1_arr[:, 2] >= z_bot) & (p1_arr[:, 2] <= z_top)
  if np.any(horiz_in_z):
    t_z_enter[horiz_in_z] = 0.0
    t_z_exit[horiz_in_z] = 1.0

  ox = p1_arr[:, 0] - params.x_kiln_center
  oy = p1_arr[:, 1] - params.y_kiln_center
  dx = p2[:, 0] - p1_arr[:, 0]
  dy = p2[:, 1] - p1_arr[:, 1]

  a_quad = dx * dx + dy * dy
  b_quad = 2.0 * (ox * dx + oy * dy)
  c_quad = ox * ox + oy * oy - (r_cyl * r_cyl)
  disc = b_quad * b_quad - 4.0 * a_quad * c_quad

  t_r_enter = np.ones(n_seg, dtype=np.float64)
  t_r_exit = np.zeros(n_seg, dtype=np.float64)

  valid_quad = (a_quad > 1e-12) & (disc > 0.0)
  if np.any(valid_quad):
    sqrt_disc = np.sqrt(np.maximum(0.0, disc[valid_quad]))
    inv_2a = 0.5 / a_quad[valid_quad]
    tr1 = (-b_quad[valid_quad] - sqrt_disc) * inv_2a
    tr2 = (-b_quad[valid_quad] + sqrt_disc) * inv_2a
    t_r_enter[valid_quad] = np.clip(np.minimum(tr1, tr2), 0.0, 1.0)
    t_r_exit[valid_quad] = np.clip(np.maximum(tr1, tr2), 0.0, 1.0)

  vert_in_r = (a_quad <= 1e-12) & (c_quad < 0.0)
  if np.any(vert_in_r):
    t_r_enter[vert_in_r] = 0.0
    t_r_exit[vert_in_r] = 1.0

  t_start = np.maximum(t_z_enter, t_r_enter)
  t_end = np.minimum(t_z_exit, t_r_exit)
  return t_end > (t_start + 1e-6)


def _is_segment_blocked_by_hood(
    p1: np.ndarray,
    p2: np.ndarray,
    params: params_lib.SimParams,
) -> np.ndarray:
  """Returns boolean mask of line segments p1 -> p2 occluded by the hood frustum.

  Models the solid conical/frustum exhaust hood centered at (x_hood, y_hood)
  spanning z in [z_hood_inlet, h_wall] with radius varying linearly from
  r_hood_inlet at z_hood_inlet to 0.5 * d_hood_duct at h_wall.
  """
  z_bot = params.z_hood_inlet + 1e-5
  z_top = params.h_wall - 1e-5
  r_bot = max(0.0, params.r_hood_inlet - 1e-4)
  r_top = max(0.0, 0.5 * params.d_hood_duct - 1e-4)
  if (
      params.z_hood_drop <= 1e-5
      or z_top <= z_bot
      or max(r_bot, r_top) <= 0.0
  ):
    return np.zeros(p2.shape[0], dtype=bool)

  p1_arr = np.broadcast_to(p1, p2.shape)
  n_seg = p2.shape[0]
  dz = p2[:, 2] - p1_arr[:, 2]

  u_lo = np.ones(n_seg, dtype=np.float64)
  u_hi = np.zeros(n_seg, dtype=np.float64)
  non_horiz = np.abs(dz) > 1e-9
  if np.any(non_horiz):
    inv_dz = 1.0 / dz[non_horiz]
    tz1 = (z_bot - p1_arr[non_horiz, 2]) * inv_dz
    tz2 = (z_top - p1_arr[non_horiz, 2]) * inv_dz
    u_lo[non_horiz] = np.clip(np.minimum(tz1, tz2), 0.0, 1.0)
    u_hi[non_horiz] = np.clip(np.maximum(tz1, tz2), 0.0, 1.0)

  horiz_in_z = (~non_horiz) & (p1_arr[:, 2] >= z_bot) & (p1_arr[:, 2] <= z_top)
  if np.any(horiz_in_z):
    u_lo[horiz_in_z] = 0.0
    u_hi[horiz_in_z] = 1.0

  u_a = np.minimum(1.0, u_lo + 1e-6)
  u_b = np.maximum(0.0, u_hi - 1e-6)
  valid_z = u_b > u_a
  if not np.any(valid_z):
    return np.zeros(n_seg, dtype=bool)

  slope = (r_top - r_bot) / params.z_hood_drop
  ox = p1_arr[:, 0] - params.x_hood
  oy = p1_arr[:, 1] - params.y_hood
  dx = p2[:, 0] - p1_arr[:, 0]
  dy = p2[:, 1] - p1_arr[:, 1]
  r0 = r_bot + slope * (p1_arr[:, 2] - params.z_hood_inlet)
  dr = slope * dz

  a_q = dx * dx + dy * dy - dr * dr
  b_q = 2.0 * (ox * dx + oy * dy - r0 * dr)
  c_q = ox * ox + oy * oy - r0 * r0

  q_a = a_q * u_a * u_a + b_q * u_a + c_q
  q_b = a_q * u_b * u_b + b_q * u_b + c_q
  q_min = np.minimum(q_a, q_b)

  # Convex parabola (A > 0) may attain its minimum strictly inside [u_a, u_b].
  convex = valid_z & (a_q > 1e-12)
  if np.any(convex):
    u_star = np.clip(
        -0.5 * b_q[convex] / a_q[convex], u_a[convex], u_b[convex]
    )
    q_star = a_q[convex] * u_star * u_star + b_q[convex] * u_star + c_q[convex]
    q_min[convex] = np.minimum(q_min[convex], q_star)

  return valid_z & (q_min < 0.0)


def _is_segment_blocked_by_solids(
    p1: np.ndarray,
    p2: np.ndarray,
    params: params_lib.SimParams,
    check_kiln_cyl: bool = True,
    check_hood: bool = True,
    check_reflector_disk: bool = False,
) -> np.ndarray:
  """Returns boolean mask of segments blocked by enabled solid bodies."""
  blocked = np.zeros(p2.shape[0], dtype=bool)
  if check_kiln_cyl:
    blocked |= _is_segment_blocked_by_kiln_cylinder(
        p1=p1, p2=p2, params=params
    )
  if check_hood:
    blocked |= _is_segment_blocked_by_hood(p1=p1, p2=p2, params=params)
  if check_reflector_disk:
    blocked |= _is_segment_blocked_by_reflector_disk(
        p1=p1, p2=p2, params=params
    )
  return blocked


def _sample_disk_quadrature(
    x_center: float,
    y_center: float,
    z_plane: float,
    r_inner: float,
    r_outer: float,
    n_r: int = _KILN_DISK_QUAD_R,
    n_theta: int = _KILN_DISK_QUAD_THETA,
) -> tuple[np.ndarray, np.ndarray, float]:
  """Samples equal-area polar midpoint elements on an annular or full disk."""
  if r_outer <= r_inner + 1e-9 or n_r <= 0 or n_theta <= 0:
    return (
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        0.0,
    )
  k_idx = np.arange(n_r, dtype=np.float64) + 0.5
  r_vals = np.sqrt(
      r_inner * r_inner
      + (r_outer * r_outer - r_inner * r_inner) * (k_idx / n_r)
  )
  theta_vals = (
      2.0 * math.pi * (np.arange(n_theta, dtype=np.float64) + 0.5) / n_theta
  )
  rr, tt = np.meshgrid(r_vals, theta_vals, indexing="ij")
  rr_flat = rr.ravel()
  tt_flat = tt.ravel()
  n_sub = rr_flat.size
  d_area = math.pi * (r_outer * r_outer - r_inner * r_inner) / n_sub
  sub_coords = np.column_stack([
      x_center + rr_flat * np.cos(tt_flat),
      y_center + rr_flat * np.sin(tt_flat),
      np.full(n_sub, z_plane, dtype=np.float64),
  ])
  return sub_coords, rr_flat, d_area


def _compute_point_to_disk_vf(
    pt_coord: np.ndarray,
    pt_normal: np.ndarray,
    disk_sub_coords: np.ndarray,
    d_area_sub: float,
    disk_nz: float,
    params: params_lib.SimParams,
    check_kiln_cyl: bool = False,
    check_reflector_disk: bool = False,
    check_hood: bool = True,
) -> float:
  """Integrates dF_{pt -> disk} over polar sub-elements with ray occlusion."""
  if disk_sub_coords.shape[0] == 0 or d_area_sub <= 0.0:
    return 0.0

  r_vec = disk_sub_coords - pt_coord[None, :]
  dist_sq = np.sum(r_vec * r_vec, axis=1)
  dist = np.sqrt(np.maximum(dist_sq, 1e-12))
  cos_pt = np.sum(r_vec * pt_normal[None, :], axis=1) / dist
  cos_disk = (-r_vec[:, 2] * disk_nz) / dist
  valid = (cos_pt > 1e-6) & (cos_disk > 1e-6)
  if not np.any(valid):
    return 0.0

  if check_kiln_cyl or check_reflector_disk or check_hood:
    valid_idx = np.flatnonzero(valid)
    p1_b = np.broadcast_to(pt_coord, disk_sub_coords[valid_idx].shape)
    blocked = _is_segment_blocked_by_solids(
        p1=disk_sub_coords[valid_idx],
        p2=p1_b,
        params=params,
        check_kiln_cyl=check_kiln_cyl,
        check_hood=check_hood,
        check_reflector_disk=check_reflector_disk,
    )
    valid[valid_idx[blocked]] = False
    if not np.any(valid):
      return 0.0

  denom = math.pi * dist_sq[valid] + 0.5 * d_area_sub
  return float(np.sum(cos_pt[valid] * cos_disk[valid] * d_area_sub / denom))


def _compute_disk_to_patches_af(
    disk_sub_coords: np.ndarray,
    rr_flat: np.ndarray,
    d_area_sub: float,
    disk_nz: float,
    c_cand: np.ndarray,
    n_cand: np.ndarray,
    a_cand: np.ndarray,
    st_cand: np.ndarray,
    params: params_lib.SimParams,
    annulus_r_min: float | None = None,
    check_kiln_cyl: bool = False,
    check_reflector_disk: bool = False,
    check_hood: bool = True,
) -> np.ndarray:
  """Integrates exchange areas A_m * F_{m -> disk} over polar sub-elements."""
  if disk_sub_coords.shape[0] == 0 or c_cand.shape[0] == 0:
    return np.zeros(c_cand.shape[0], dtype=np.float64)

  r_vec_mr = disk_sub_coords[None, :, :] - c_cand[:, None, :]
  dist_sq_mr = np.sum(r_vec_mr * r_vec_mr, axis=2)
  dist_mr = np.sqrt(np.maximum(dist_sq_mr, 1e-12))
  cos_m_d = np.sum(r_vec_mr * n_cand[:, None, :], axis=2) / dist_mr
  cos_disk = (-r_vec_mr[:, :, 2] * disk_nz) / dist_mr
  vis_mr = (cos_m_d > 1e-6) & (cos_disk > 1e-6)

  if annulus_r_min is not None:
    annulus_only_rows = (
        (st_cand == int(mesh_lib.SurfaceType.KILN_SIDE))
        | (st_cand == int(mesh_lib.SurfaceType.SSR_BOX))
    )
    if np.any(annulus_only_rows):
      vis_mr[annulus_only_rows, :] &= rr_flat[None, :] >= annulus_r_min - 1e-4

  if check_kiln_cyl or check_reflector_disk or check_hood:
    vis_m_idx, vis_q_idx = np.nonzero(vis_mr)
    if vis_m_idx.size > 0:
      p1_vis = disk_sub_coords[vis_q_idx]
      p2_vis = c_cand[vis_m_idx]
      blocked = _is_segment_blocked_by_solids(
          p1=p1_vis,
          p2=p2_vis,
          params=params,
          check_kiln_cyl=check_kiln_cyl,
          check_hood=check_hood,
          check_reflector_disk=check_reflector_disk,
      )
      vis_mr[vis_m_idx[blocked], vis_q_idx[blocked]] = False

  denom_mr = math.pi * dist_sq_mr + 0.5 * (a_cand[:, None] + d_area_sub)
  af_sub = np.where(
      vis_mr,
      cos_m_d * cos_disk * a_cand[:, None] * d_area_sub / denom_mr,
      0.0,
  )
  return np.sum(af_sub, axis=1)


def _compute_point_to_kiln_bottom_vf_with_occlusion(
    pt_coord: np.ndarray,
    pt_normal: np.ndarray,
    params: params_lib.SimParams,
) -> float:
  """Integrates the view factor from a floor point to KILN_BOTTOM with occlusion."""
  if params.r_kiln <= 0.0 or pt_coord[2] >= params.z_drum_bot - 1e-6:
    return 0.0
  sub_coords, _, d_area = _sample_disk_quadrature(
      x_center=params.x_kiln_center,
      y_center=params.y_kiln_center,
      z_plane=params.z_drum_bot,
      r_inner=0.0,
      r_outer=params.r_kiln,
  )
  return _compute_point_to_disk_vf(
      pt_coord=pt_coord,
      pt_normal=pt_normal,
      disk_sub_coords=sub_coords,
      d_area_sub=d_area,
      disk_nz=-1.0,
      params=params,
      check_reflector_disk=True,
  )


def _compute_point_to_patches_vf(
    pt_coord: np.ndarray,
    pt_normal: np.ndarray,
    patch_centroids: np.ndarray,
    patch_normals: np.ndarray,
    patch_areas: np.ndarray,
    exclude_mask: np.ndarray,
    params: params_lib.SimParams | None = None,
    check_reflector_disk: bool = True,
) -> np.ndarray:
  """Computes differential view factors dF_{pt -> n} from a point to patches."""
  r_vec = patch_centroids - pt_coord[None, :]
  dist_sq = np.sum(r_vec * r_vec, axis=1)
  dist = np.sqrt(np.maximum(dist_sq, 1e-12))

  cos_theta_pt = np.sum(r_vec * pt_normal[None, :], axis=1) / dist
  cos_theta_n = -np.sum(r_vec * patch_normals, axis=1) / dist

  valid = (cos_theta_pt > 1e-6) & (cos_theta_n > 1e-6) & (~exclude_mask)
  if params is not None and np.any(valid):
    valid_idx = np.flatnonzero(valid)
    blocked = _is_segment_blocked_by_solids(
        p1=pt_coord,
        p2=patch_centroids[valid_idx],
        params=params,
        check_kiln_cyl=True,
        check_hood=True,
        check_reflector_disk=check_reflector_disk,
    )
    valid[valid_idx[blocked]] = False

  vf = np.zeros_like(patch_areas, dtype=np.float64)
  denom = math.pi * dist_sq[valid] + 0.5 * patch_areas[valid]
  vf[valid] = (
      cos_theta_pt[valid] * cos_theta_n[valid] * patch_areas[valid] / denom
  )
  return vf


def compute_view_factors(
    mesh: mesh_lib.SurfaceMesh,
    params: params_lib.SimParams,
) -> ViewFactorSystem:
  """Computes the N x N effective view factor matrix and point sensor vectors."""
  n_patches = mesh.num_patches
  c = mesh.centroids
  n_vec = mesh.normals
  areas = mesh.areas
  stypes = mesh.surface_types

  # 1. Pairwise displacement vectors: r_vec[m, n] = c[n] - c[m]
  r_vec = c[None, :, :] - c[:, None, :]
  dist_sq = np.sum(r_vec * r_vec, axis=2)
  np.fill_diagonal(dist_sq, 1.0)
  dist = np.sqrt(dist_sq)

  cos_m = np.sum(r_vec * n_vec[:, None, :], axis=2) / dist
  cos_n = -np.sum(r_vec * n_vec[None, :, :], axis=2) / dist

  vis = (cos_m > 1e-6) & (cos_n > 1e-6)
  np.fill_diagonal(vis, False)

  for planar_type in (
      mesh_lib.SurfaceType.SHIELD_LEFT,
      mesh_lib.SurfaceType.SHIELD_BACK,
      mesh_lib.SurfaceType.GYPSUM_LEFT,
      mesh_lib.SurfaceType.GYPSUM_BACK,
      mesh_lib.SurfaceType.KILN_LID,
      mesh_lib.SurfaceType.KILN_BOTTOM,
      mesh_lib.SurfaceType.REFLECTOR_TOP,
      mesh_lib.SurfaceType.SSR_BOX,
  ):
    mask = stypes == int(planar_type)
    vis[np.ix_(mask, mask)] = False

  kiln_side_mask = stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
  ssr_mask = stypes == int(mesh_lib.SurfaceType.SSR_BOX)
  vis[np.ix_(kiln_side_mask, ssr_mask)] = False
  vis[np.ix_(ssr_mask, kiln_side_mask)] = False

  # Block pairwise diffuse rays occluded by solid kiln drum, exhaust hood,
  # or floor reflector disk.
  vis_m_idx, vis_n_idx = np.nonzero(np.triu(vis, k=1))
  if vis_m_idx.size > 0:
    blocked_vis = _is_segment_blocked_by_solids(
        p1=c[vis_m_idx],
        p2=c[vis_n_idx],
        params=params,
        check_kiln_cyl=True,
        check_hood=True,
        check_reflector_disk=True,
    )
    if np.any(blocked_vis):
      bm = vis_m_idx[blocked_vis]
      bn = vis_n_idx[blocked_vis]
      vis[bm, bn] = False
      vis[bn, bm] = False

  # 2. Symmetric exchange area matrix A_m * F_{m -> n}
  af_matrix = np.zeros((n_patches, n_patches), dtype=np.float64)
  denom_matrix = math.pi * dist_sq + 0.5 * (areas[:, None] + areas[None, :])
  af_matrix[vis] = (
      (cos_m[vis] * cos_n[vis] / denom_matrix[vis])
      * (areas[:, None] * areas[None, :])[vis]
  )

  # 3. Override coaxial disk pair (KILN_BOTTOM <-> REFLECTOR_TOP)
  bot_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
  lid_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.KILN_LID))
  ref_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP))
  ssr_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.SSR_BOX))
  if bot_ids.size == 1 and ref_ids.size == 1:
    b_id = int(bot_ids[0])
    r_id = int(ref_ids[0])
    vf_bot_to_ref = coaxial_parallel_disks_vf(
        r1=params.r_kiln,
        r2=params.r_reflector,
        h=params.z_drum_bot - params.z_reflector_top,
    )
    af_exact = areas[b_id] * vf_bot_to_ref
    af_matrix[b_id, r_id] = af_exact
    af_matrix[r_id, b_id] = af_exact

  # 3b. Finite-disk diffuse exchange between REFLECTOR_TOP and all non-bottom
  # patches above z_reflector_top, accounting for occlusion by the solid kiln
  # cylinder (r <= r_kiln, z in [z_drum_bot, z_kiln_top]).
  if ref_ids.size == 1 and params.r_reflector > 0.0:
    r_id = int(ref_ids[0])
    b_id_exclusion = int(bot_ids[0]) if bot_ids.size == 1 else -1
    for m_idx in range(n_patches):
      if m_idx != b_id_exclusion and m_idx != r_id:
        af_matrix[m_idx, r_id] = 0.0
        af_matrix[r_id, m_idx] = 0.0

    ref_sub_coords, rr_ref_flat, d_area_ref = _sample_disk_quadrature(
        x_center=params.x_kiln_center,
        y_center=params.y_kiln_center,
        z_plane=params.z_reflector_top,
        r_inner=0.0,
        r_outer=params.r_reflector,
    )
    cand_mask_ref = (
        (c[:, 2] > params.z_reflector_top + 1e-4)
        & (stypes != int(mesh_lib.SurfaceType.REFLECTOR_TOP))
        & (stypes != int(mesh_lib.SurfaceType.KILN_BOTTOM))
        & (stypes != int(mesh_lib.SurfaceType.KILN_LID))
    )
    cand_ids_ref = np.flatnonzero(cand_mask_ref)
    if cand_ids_ref.size > 0:
      af_cand_ref = _compute_disk_to_patches_af(
          disk_sub_coords=ref_sub_coords,
          rr_flat=rr_ref_flat,
          d_area_sub=d_area_ref,
          disk_nz=1.0,
          c_cand=c[cand_ids_ref],
          n_cand=n_vec[cand_ids_ref],
          a_cand=areas[cand_ids_ref],
          st_cand=stypes[cand_ids_ref],
          params=params,
          annulus_r_min=params.r_kiln,
          check_kiln_cyl=True,
      )
      af_matrix[cand_ids_ref, r_id] = af_cand_ref
      af_matrix[r_id, cand_ids_ref] = af_cand_ref

  # 3c. Finite-disk diffuse exchange between KILN_LID and patches above lid.
  if lid_ids.size == 1 and params.r_kiln > 0.0:
    lid_id = int(lid_ids[0])
    af_matrix[:, lid_id] = 0.0
    af_matrix[lid_id, :] = 0.0
    lid_sub_coords, rr_lid_flat, d_area_lid = _sample_disk_quadrature(
        x_center=params.x_kiln_center,
        y_center=params.y_kiln_center,
        z_plane=params.z_kiln_top,
        r_inner=0.0,
        r_outer=params.r_kiln,
    )
    cand_mask_lid = (c[:, 2] > params.z_kiln_top + 1e-4) & (
        stypes != int(mesh_lib.SurfaceType.KILN_LID)
    )
    cand_ids_lid = np.flatnonzero(cand_mask_lid)
    if cand_ids_lid.size > 0:
      af_cand_lid = _compute_disk_to_patches_af(
          disk_sub_coords=lid_sub_coords,
          rr_flat=rr_lid_flat,
          d_area_sub=d_area_lid,
          disk_nz=1.0,
          c_cand=c[cand_ids_lid],
          n_cand=n_vec[cand_ids_lid],
          a_cand=areas[cand_ids_lid],
          st_cand=stypes[cand_ids_lid],
          params=params,
      )
      af_matrix[cand_ids_lid, lid_id] = af_cand_lid
      af_matrix[lid_id, cand_ids_lid] = af_cand_lid

  # 3d. Finite-disk diffuse exchange between KILN_BOTTOM and non-reflector
  # patches below z_drum_bot, accounting for occlusion by the floor reflector.
  if bot_ids.size == 1 and params.r_kiln > 0.0:
    b_id = int(bot_ids[0])
    r_id_exclusion = int(ref_ids[0]) if ref_ids.size == 1 else -1
    for m_idx in range(n_patches):
      if m_idx != r_id_exclusion and m_idx != b_id:
        af_matrix[m_idx, b_id] = 0.0
        af_matrix[b_id, m_idx] = 0.0

    bot_sub_coords, rr_bot_flat, d_area_bot = _sample_disk_quadrature(
        x_center=params.x_kiln_center,
        y_center=params.y_kiln_center,
        z_plane=params.z_drum_bot,
        r_inner=0.0,
        r_outer=params.r_kiln,
    )
    cand_mask_bot = (
        (c[:, 2] < params.z_drum_bot - 1e-4)
        & (stypes != int(mesh_lib.SurfaceType.KILN_BOTTOM))
        & (stypes != int(mesh_lib.SurfaceType.REFLECTOR_TOP))
    )
    cand_ids_bot = np.flatnonzero(cand_mask_bot)
    if cand_ids_bot.size > 0:
      af_cand_bot = _compute_disk_to_patches_af(
          disk_sub_coords=bot_sub_coords,
          rr_flat=rr_bot_flat,
          d_area_sub=d_area_bot,
          disk_nz=-1.0,
          c_cand=c[cand_ids_bot],
          n_cand=n_vec[cand_ids_bot],
          a_cand=areas[cand_ids_bot],
          st_cand=stypes[cand_ids_bot],
          params=params,
          check_reflector_disk=True,
      )
      af_matrix[cand_ids_bot, b_id] = af_cand_bot
      af_matrix[b_id, cand_ids_bot] = af_cand_bot

  # 3e. Tangential SSR_BOX side flanges <-> KILN_SIDE direct exchange via
  # surface quadrature (the rear face ssr_ids[0] points radially outward).
  kiln_side_ids_all = np.flatnonzero(kiln_side_mask)
  if (
      ssr_ids.size == 3
      and kiln_side_ids_all.size > 0
      and params.d_ssr > 0.0
      and params.h_ssr > 0.0
  ):
    ur_x = math.cos(params.ssr_angle_rad)
    ur_y = math.sin(params.ssr_angle_rad)
    ut_x = -ur_y
    ut_y = ur_x
    n_u = constants.SSR_QUAD_U
    n_v = constants.SSR_QUAD_V
    u_fracs = (np.arange(n_u, dtype=np.float64) + 0.5) / n_u
    v_fracs = (np.arange(n_v, dtype=np.float64) + 0.5) / n_v
    rr_f, zz_f = np.meshgrid(
        params.r_kiln + params.d_ssr * u_fracs,
        params.z_ssr_bot + params.h_ssr * v_fracs,
        indexing="ij",
    )
    rr_flat = rr_f.ravel()
    zz_flat = zz_f.ravel()
    d_area_ssr = (params.d_ssr * params.h_ssr) / (n_u * n_v)
    c_ks = c[kiln_side_ids_all]
    n_ks = n_vec[kiln_side_ids_all]
    a_ks = areas[kiln_side_ids_all]

    for flange_id, sign in ((int(ssr_ids[1]), -1.0), (int(ssr_ids[2]), 1.0)):
      sub_pts = np.column_stack([
          params.x_kiln_center
          + rr_flat * ur_x
          + sign * 0.5 * params.w_ssr * ut_x,
          params.y_kiln_center
          + rr_flat * ur_y
          + sign * 0.5 * params.w_ssr * ut_y,
          zz_flat,
      ])
      flange_normal = np.array([sign * ut_x, sign * ut_y, 0.0], dtype=np.float64)
      r_fk = c_ks[None, :, :] - sub_pts[:, None, :]
      d_sq_fk = np.sum(r_fk * r_fk, axis=2)
      d_fk = np.sqrt(np.maximum(d_sq_fk, 1e-12))
      cos_f = np.sum(r_fk * flange_normal[None, None, :], axis=2) / d_fk
      cos_k = -np.sum(r_fk * n_ks[None, :, :], axis=2) / d_fk
      vis_fk = (cos_f > 1e-6) & (cos_k > 1e-6)
      q_idx, k_idx = np.nonzero(vis_fk)
      if q_idx.size > 0:
        blocked_fk = _is_segment_blocked_by_solids(
            p1=sub_pts[q_idx],
            p2=c_ks[k_idx],
            params=params,
            check_kiln_cyl=True,
            check_hood=True,
            check_reflector_disk=True,
        )
        vis_fk[q_idx[blocked_fk], k_idx[blocked_fk]] = False
      denom_fk = math.pi * d_sq_fk + 0.5 * (a_ks[None, :] + d_area_ssr)
      af_fk = np.sum(
          np.where(
              vis_fk,
              cos_f * cos_k * d_area_ssr * a_ks[None, :] / denom_fk,
              0.0,
          ),
          axis=0,
      )
      af_matrix[flange_id, kiln_side_ids_all] = af_fk
      af_matrix[kiln_side_ids_all, flange_id] = af_fk

  f_direct = np.divide(
      af_matrix,
      areas[:, None],
      out=np.zeros_like(af_matrix),
      where=areas[:, None] > 0.0,
  )
  dir_row_sums = np.sum(f_direct, axis=1)
  if np.any(dir_row_sums > constants.MAX_PATCH_VF_ROW_SUM):
    s_dir = np.where(
        dir_row_sums > constants.MAX_PATCH_VF_ROW_SUM,
        constants.MAX_PATCH_VF_ROW_SUM / np.maximum(dir_row_sums, 1e-12),
        1.0,
    )
    af_matrix *= np.minimum(s_dir[:, None], s_dir[None, :])
    f_direct = np.divide(
        af_matrix,
        areas[:, None],
        out=np.zeros_like(af_matrix),
        where=areas[:, None] > 0.0,
    )

  # 4. Specular reflection from the circular floor reflector (REFLECTOR_TOP)
  # for both the finite KILN_BOTTOM disk and KILN_SIDE patches onto walls
  # (SHIELD_LEFT, SHIELD_BACK, GYPSUM_LEFT, GYPSUM_BACK) and SSR_BOX.
  f_mirror = np.zeros((n_patches, n_patches), dtype=np.float64)
  if params.rho_reflector_top > 0.0 and params.r_reflector > 0.0:
    target_mask = (
        (c[:, 2] > params.z_reflector_top + 1e-4)
        & (stypes != int(mesh_lib.SurfaceType.REFLECTOR_TOP))
        & (stypes != int(mesh_lib.SurfaceType.KILN_BOTTOM))
        & (stypes != int(mesh_lib.SurfaceType.KILN_LID))
    )
    target_ids = np.flatnonzero(target_mask)

    # 4a. Specular reflection of the finite KILN_BOTTOM disk (at z = z_mirror)
    if bot_ids.size == 1:
      b_id = int(bot_ids[0])
      img_coords_b, rr_b_flat, d_area_b = _sample_disk_quadrature(
          x_center=params.x_kiln_center,
          y_center=params.y_kiln_center,
          z_plane=params.z_mirror,
          r_inner=0.0,
          r_outer=params.r_kiln,
      )
      r_vals = rr_b_flat.reshape(
          _KILN_DISK_QUAD_R, _KILN_DISK_QUAD_THETA
      )[:, 0]

      # Specular self-return F_{KILN_BOTTOM -> KILN_BOTTOM | reflector}.
      dz_bot_ref = params.z_drum_bot - params.z_reflector_top
      if dz_bot_ref > 1e-6:
        if params.r_reflector >= params.r_kiln - 1e-6:
          vf_bb_mirror = coaxial_parallel_disks_vf(
              r1=params.r_kiln,
              r2=params.r_kiln,
              h=2.0 * dz_bot_ref,
          )
        else:
          src_x_1d = params.x_kiln_center + r_vals
          src_y_1d = np.full(
              _KILN_DISK_QUAD_R, params.y_kiln_center, dtype=np.float64
          )
          dx_bb = img_coords_b[None, :, 0] - src_x_1d[:, None]
          dy_bb = img_coords_b[None, :, 1] - src_y_1d[:, None]
          dz_bb = 2.0 * dz_bot_ref
          d_sq_bb = dx_bb * dx_bb + dy_bb * dy_bb + dz_bb * dz_bb
          cos_bb = dz_bb / np.sqrt(np.maximum(d_sq_bb, 1e-12))
          x_mid_bb = 0.5 * (src_x_1d[:, None] + img_coords_b[None, :, 0])
          y_mid_bb = 0.5 * (src_y_1d[:, None] + img_coords_b[None, :, 1])
          r_mid_bb = np.hypot(
              x_mid_bb - params.x_kiln_center,
              y_mid_bb - params.y_kiln_center,
          )
          hit_bb = r_mid_bb <= params.r_reflector + 1e-4
          vf_ring = np.sum(
              np.where(
                  hit_bb,
                  (cos_bb * cos_bb * d_area_b) / (math.pi * d_sq_bb + d_area_b),
                  0.0,
              ),
              axis=1,
          )
          vf_bb_mirror = float(np.mean(vf_ring))
        if ref_ids.size == 1:
          vf_bb_mirror = min(vf_bb_mirror, float(f_direct[b_id, int(ref_ids[0])]))
        f_mirror[b_id, b_id] += params.rho_reflector_top * vf_bb_mirror

      if target_ids.size > 0:
        c_t = c[target_ids]
        n_t = n_vec[target_ids]
        a_t = areas[target_ids]
        st_t = stypes[target_ids]

        # Vector from target patch m to virtual image sub-element q: (M_t, N_sub, 3)
        r_t_to_b = img_coords_b[None, :, :] - c_t[:, None, :]
        d_sq_tb = np.sum(r_t_to_b * r_t_to_b, axis=2)
        d_tb = np.sqrt(np.maximum(d_sq_tb, 1e-12))

        cos_t = np.sum(r_t_to_b * n_t[:, None, :], axis=2) / d_tb
        cos_img_b = -r_t_to_b[:, :, 2] / d_tb

        dz_tb = c_t[:, 2] - params.z_mirror
        t_hit_b = (c_t[:, 2] - params.z_reflector_top) / np.maximum(
            dz_tb, 1e-6
        )
        x_hit_b = c_t[:, 0, None] + t_hit_b[:, None] * (
            img_coords_b[None, :, 0] - c_t[:, 0, None]
        )
        y_hit_b = c_t[:, 1, None] + t_hit_b[:, None] * (
            img_coords_b[None, :, 1] - c_t[:, 1, None]
        )
        r_hit_b = np.hypot(
            x_hit_b - params.x_kiln_center,
            y_hit_b - params.y_kiln_center,
        )

        hit_b = (
            (cos_t > 1e-6)
            & (cos_img_b > 1e-6)
            & (r_hit_b <= params.r_reflector + 1e-4)
        )
        # SSR_BOX and KILN_SIDE sit outside the kiln footprint and can only see
        # reflections off the overhanging annulus (r_hit >= r_kiln).
        ann_only_rows = (
            (st_t == int(mesh_lib.SurfaceType.SSR_BOX))
            | (st_t == int(mesh_lib.SurfaceType.KILN_SIDE))
        )
        if np.any(ann_only_rows):
          hit_b[ann_only_rows, :] &= (
              r_hit_b[ann_only_rows, :] >= params.r_kiln - 1e-4
          )

        # Block any reflected ray whose path from the reflector hit point to the
        # target patch intersects the solid kiln drum (including near-side rays
        # that rise into the underside of KILN_BOTTOM before clearing r_kiln).
        hit_m_idx, hit_q_idx = np.nonzero(hit_b)
        if hit_m_idx.size > 0:
          p1_hits = np.column_stack([
              x_hit_b[hit_m_idx, hit_q_idx],
              y_hit_b[hit_m_idx, hit_q_idx],
              np.full(hit_m_idx.size, params.z_reflector_top, dtype=np.float64),
          ])
          p2_targets = c_t[hit_m_idx]
          blocked_b = _is_segment_blocked_by_solids(
              p1=p1_hits,
              p2=p2_targets,
              params=params,
              check_kiln_cyl=True,
              check_hood=True,
              check_reflector_disk=False,
          )
          hit_b[hit_m_idx[blocked_b], hit_q_idx[blocked_b]] = False

        denom_tb = math.pi * d_sq_tb + 0.5 * (a_t[:, None] + d_area_b)
        vf_tb_sub = np.where(
            hit_b, cos_t * cos_img_b * d_area_b / denom_tb, 0.0
        )
        vf_t_to_b = np.sum(vf_tb_sub, axis=1)

        f_mirror[target_ids, b_id] += params.rho_reflector_top * vf_t_to_b
        f_mirror[b_id, target_ids] += (
            params.rho_reflector_top * vf_t_to_b * a_t / areas[b_id]
        )

    # 4b. Specular reflection of KILN_SIDE patches off the overhanging
    # reflector annulus (r in [r_kiln, r_reflector]) onto walls and SSR_BOX.
    kiln_side_ids = np.flatnonzero(
        stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
    )
    wall_target_ids = target_ids[
        stypes[target_ids] != int(mesh_lib.SurfaceType.KILN_SIDE)
    ]
    if (
        kiln_side_ids.size > 0
        and wall_target_ids.size > 0
        and params.r_reflector > params.r_kiln + 1e-4
    ):
      c_ks = c[kiln_side_ids]
      n_ks = n_vec[kiln_side_ids]
      a_ks = areas[kiln_side_ids]

      c_ks_img = c_ks.copy()
      c_ks_img[:, 2] = 2.0 * params.z_reflector_top - c_ks[:, 2]
      n_ks_img = n_ks.copy()
      n_ks_img[:, 2] = -n_ks[:, 2]

      c_t = c[wall_target_ids]
      n_t = n_vec[wall_target_ids]
      a_t = areas[wall_target_ids]

      # Vector from target patch m to KILN_SIDE virtual image k: (M_t, N_ks, 3)
      r_t_to_ks = c_ks_img[None, :, :] - c_t[:, None, :]
      d_sq_tks = np.sum(r_t_to_ks * r_t_to_ks, axis=2)
      d_tks = np.sqrt(np.maximum(d_sq_tks, 1e-12))

      cos_t_ks = np.sum(r_t_to_ks * n_t[:, None, :], axis=2) / d_tks
      cos_ks = -np.sum(r_t_to_ks * n_ks_img[None, :, :], axis=2) / d_tks

      dz_tks = c_t[:, 2, None] - c_ks_img[None, :, 2]
      valid_tks = (c_ks[None, :, 2] > params.z_reflector_top + 1e-4) & (
          dz_tks > 1e-6
      )
      t_hit_ks = np.where(
          valid_tks,
          (c_t[:, 2, None] - params.z_reflector_top)
          / np.maximum(dz_tks, 1e-6),
          0.0,
      )
      x_hit_ks = c_t[:, 0, None] + t_hit_ks * (
          c_ks_img[None, :, 0] - c_t[:, 0, None]
      )
      y_hit_ks = c_t[:, 1, None] + t_hit_ks * (
          c_ks_img[None, :, 1] - c_t[:, 1, None]
      )
      r_hit_ks = np.hypot(
          x_hit_ks - params.x_kiln_center,
          y_hit_ks - params.y_kiln_center,
      )

      hit_ks = (
          valid_tks
          & (cos_t_ks > 1e-6)
          & (cos_ks > 1e-6)
          & (r_hit_ks >= params.r_kiln - 1e-4)
          & (r_hit_ks <= params.r_reflector + 1e-4)
      )
      ks_m_idx, ks_k_idx = np.nonzero(hit_ks)
      if ks_m_idx.size > 0:
        p1_ks_hits = np.column_stack([
            x_hit_ks[ks_m_idx, ks_k_idx],
            y_hit_ks[ks_m_idx, ks_k_idx],
            np.full(ks_m_idx.size, params.z_reflector_top, dtype=np.float64),
        ])
        p2_ks_targets = c_t[ks_m_idx]
        blocked_ks = _is_segment_blocked_by_solids(
            p1=p1_ks_hits,
            p2=p2_ks_targets,
            params=params,
            check_kiln_cyl=True,
            check_hood=True,
            check_reflector_disk=False,
        )
        hit_ks[ks_m_idx[blocked_ks], ks_k_idx[blocked_ks]] = False

      denom_tks = math.pi * d_sq_tks + 0.5 * (a_t[:, None] + a_ks[None, :])
      vf_t_to_ks = np.where(
          hit_ks, cos_t_ks * cos_ks * a_ks[None, :] / denom_tks, 0.0
      )
      f_mirror[np.ix_(wall_target_ids, kiln_side_ids)] += (
          params.rho_reflector_top * vf_t_to_ks
      )
      f_mirror[np.ix_(kiln_side_ids, wall_target_ids)] += (
          params.rho_reflector_top
          * vf_t_to_ks.T
          * (a_t[None, :] / a_ks[:, None])
      )

  # Assemble exact Oppenheim diffuse-specular matrix:
  # - Mirror surface r_id does not reflect diffusely (J[r_id] = E_b[r_id]), so
  #   incoming direct rays absorbed by r_id are weighted by eps_reflector_top.
  # - Symmetric scaling of f_mirror preserves A_m * F_mirror[m,n] = A_n * F_mirror[n,m]
  #   while ensuring every row sums to <= MAX_PATCH_VF_ROW_SUM.
  if ref_ids.size == 1 and params.r_reflector > 0.0:
    r_id = int(ref_ids[0])
    direct_sum = np.sum(f_direct, axis=1) - (
        (1.0 - params.eps_reflector_top) * f_direct[:, r_id]
    )
    avail_m = np.maximum(0.0, constants.MAX_PATCH_VF_ROW_SUM - direct_sum)
    mir_sum_m = np.sum(f_mirror, axis=1)
    if np.any(mir_sum_m > avail_m):
      s_m = np.where(
          mir_sum_m > avail_m,
          avail_m / np.maximum(mir_sum_m, 1e-12),
          1.0,
      )
      f_mirror *= np.minimum(s_m[:, None], s_m[None, :])

    f_eff = f_direct + f_mirror
    f_eff[:, r_id] = params.eps_reflector_top * f_direct[:, r_id]
    f_eff[r_id, :] = f_direct[r_id, :].copy()
    f_eff[r_id, r_id] = 0.0
  else:
    f_eff = f_direct.copy()

  # 5. Enforce row sum <= MAX_PATCH_VF_ROW_SUM (guard against floating-point overshoot)
  row_sums = np.sum(f_eff, axis=1)
  over_mask = row_sums > constants.MAX_PATCH_VF_ROW_SUM
  if np.any(over_mask):
    scale = constants.MAX_PATCH_VF_ROW_SUM / row_sums[over_mask]
    f_eff[over_mask, :] *= scale[:, None]
    row_sums = np.sum(f_eff, axis=1)

  f_room = np.maximum(0.0, 1.0 - row_sums)

  # 6. Floor point-to-patch view factor vectors (accounting for occlusion by
  # the circular reflector disk, kiln cylinder, hood frustum, and finite-disk
  # integration over KILN_BOTTOM).
  sensor_vf: dict[str, np.ndarray] = {}
  exclude_floor = (
      (stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP))
      | (stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
  )
  for name, sensor in mesh.floor_sensors.items():
    pt_c = np.array(sensor.position_m, dtype=np.float64)
    pt_n = np.array(sensor.normal, dtype=np.float64)
    vf_vec = _compute_point_to_patches_vf(
        pt_coord=pt_c,
        pt_normal=pt_n,
        patch_centroids=c,
        patch_normals=n_vec,
        patch_areas=areas,
        exclude_mask=exclude_floor,
        params=params,
        check_reflector_disk=True,
    )
    if bot_ids.size == 1:
      vf_vec[int(bot_ids[0])] = _compute_point_to_kiln_bottom_vf_with_occlusion(
          pt_coord=pt_c,
          pt_normal=pt_n,
          params=params,
      )
    s_sum = float(np.sum(vf_vec))
    if s_sum > constants.MAX_SENSOR_VF_SUM:
      vf_vec *= constants.MAX_SENSOR_VF_SUM / s_sum
    sensor_vf[name] = vf_vec

  # 7. Ceiling downward-facing point-to-patch view factor vectors at z = h_wall
  sensor_vf.update(compute_ceiling_sensor_vfs(mesh=mesh, params=params))

  return ViewFactorSystem(
      f_eff=f_eff,
      f_room=f_room,
      sensor_vf=sensor_vf,
  )


def compute_ceiling_sensor_vfs(
    mesh: mesh_lib.SurfaceMesh,
    params: params_lib.SimParams,
) -> dict[str, np.ndarray]:
  """Computes the downward-facing ceiling sensor view factor vectors at z = h_wall."""
  c = mesh.centroids
  n_vec = mesh.normals
  areas = mesh.areas
  stypes = mesh.surface_types
  bot_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
  lid_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.KILN_LID))
  ref_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP))

  diag_inv = 1.0 / math.sqrt(2.0)
  r_ceil_sensor = params.r_hood_inlet + CEILING_SENSOR_RIM_OFFSET_M
  ceiling_points = {
      "ceiling_hood_rim_corner": (
          max(
              CEILING_SENSOR_WALL_OFFSET_M,
              params.x_hood - r_ceil_sensor * diag_inv,
          ),
          max(
              CEILING_SENSOR_WALL_OFFSET_M,
              params.y_hood - r_ceil_sensor * diag_inv,
          ),
          params.h_wall,
      ),
      "ceiling_hood_rim_room": (
          params.x_hood + r_ceil_sensor * diag_inv,
          params.y_hood + r_ceil_sensor * diag_inv,
          params.h_wall,
      ),
      "ceiling_chimney_corner": (
          params.x_kiln_center,
          params.y_shield_front_left_wall + CEILING_SENSOR_WALL_OFFSET_M,
          params.h_wall,
      ),
  }
  pt_n_ceil = np.array([0.0, 0.0, -1.0], dtype=np.float64)
  exclude_ceil = (
      (stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
      | (stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP))
      | (stypes == int(mesh_lib.SurfaceType.KILN_LID))
  )
  kiln_side_ids_ceil = np.flatnonzero(
      stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
  )
  lid_sub_coords, _, d_area_lid = _sample_disk_quadrature(
      x_center=params.x_kiln_center,
      y_center=params.y_kiln_center,
      z_plane=params.z_kiln_top,
      r_inner=0.0,
      r_outer=params.r_kiln,
  )
  ann_coords, _, d_area_ann = _sample_disk_quadrature(
      x_center=params.x_kiln_center,
      y_center=params.y_kiln_center,
      z_plane=params.z_reflector_top,
      r_inner=params.r_kiln,
      r_outer=params.r_reflector,
  )
  img_coords_b, _, d_area_b = _sample_disk_quadrature(
      x_center=params.x_kiln_center,
      y_center=params.y_kiln_center,
      z_plane=params.z_mirror,
      r_inner=0.0,
      r_outer=params.r_kiln,
  )

  ceil_sensor_vf: dict[str, np.ndarray] = {}
  for name, coord in ceiling_points.items():
    pt_c = np.array(coord, dtype=np.float64)
    vf_vec = _compute_point_to_patches_vf(
        pt_coord=pt_c,
        pt_normal=pt_n_ceil,
        patch_centroids=c,
        patch_normals=n_vec,
        patch_areas=areas,
        exclude_mask=exclude_ceil,
        params=params,
        check_reflector_disk=True,
    )
    if lid_ids.size == 1 and params.r_kiln > 0.0:
      vf_vec[int(lid_ids[0])] = _compute_point_to_disk_vf(
          pt_coord=pt_c,
          pt_normal=pt_n_ceil,
          disk_sub_coords=lid_sub_coords,
          d_area_sub=d_area_lid,
          disk_nz=1.0,
          params=params,
      )
    if ref_ids.size == 1 and params.r_reflector > params.r_kiln + 1e-4:
      r_id = int(ref_ids[0])
      vf_ceil_ref_direct = _compute_point_to_disk_vf(
          pt_coord=pt_c,
          pt_normal=pt_n_ceil,
          disk_sub_coords=ann_coords,
          d_area_sub=d_area_ann,
          disk_nz=1.0,
          params=params,
          check_kiln_cyl=True,
      )

      if params.rho_reflector_top > 0.0:
        # (i) Specular bounce from KILN_SIDE virtual images to ceiling sensor
        if kiln_side_ids_ceil.size > 0:
          c_ks = c[kiln_side_ids_ceil]
          n_ks = n_vec[kiln_side_ids_ceil]
          a_ks = areas[kiln_side_ids_ceil]
          c_ks_img = c_ks.copy()
          c_ks_img[:, 2] = 2.0 * params.z_reflector_top - c_ks[:, 2]
          n_ks_img = n_ks.copy()
          n_ks_img[:, 2] = -n_ks[:, 2]

          r_c_to_ks = c_ks_img - pt_c[None, :]
          d_sq_cks = np.sum(r_c_to_ks * r_c_to_ks, axis=1)
          d_cks = np.sqrt(np.maximum(d_sq_cks, 1e-12))
          cos_c_ks = -r_c_to_ks[:, 2] / d_cks
          cos_ks_c = -np.sum(r_c_to_ks * n_ks_img, axis=1) / d_cks

          dz_cks = pt_c[2] - c_ks_img[:, 2]
          valid_cks = (c_ks[:, 2] > params.z_reflector_top + 1e-4) & (
              dz_cks > 1e-6
          )
          t_cks = np.where(
              valid_cks,
              (pt_c[2] - params.z_reflector_top) / np.maximum(dz_cks, 1e-6),
              0.0,
          )
          x_hit_cks = pt_c[0] + t_cks * (c_ks_img[:, 0] - pt_c[0])
          y_hit_cks = pt_c[1] + t_cks * (c_ks_img[:, 1] - pt_c[1])
          r_hit_cks = np.hypot(
              x_hit_cks - params.x_kiln_center,
              y_hit_cks - params.y_kiln_center,
          )
          hit_cks = (
              valid_cks
              & (cos_c_ks > 1e-6)
              & (cos_ks_c > 1e-6)
              & (r_hit_cks >= params.r_kiln - 1e-4)
              & (r_hit_cks <= params.r_reflector + 1e-4)
          )
          if np.any(hit_cks):
            hit_cks_idx = np.flatnonzero(hit_cks)
            p1_cks = np.column_stack([
                x_hit_cks[hit_cks_idx],
                y_hit_cks[hit_cks_idx],
                np.full(
                    hit_cks_idx.size, params.z_reflector_top, dtype=np.float64
                ),
            ])
            blocked_cks = _is_segment_blocked_by_solids(
                p1=p1_cks,
                p2=np.broadcast_to(pt_c, p1_cks.shape),
                params=params,
                check_kiln_cyl=True,
                check_hood=True,
                check_reflector_disk=False,
            )
            hit_cks[hit_cks_idx[blocked_cks]] = False
            if np.any(hit_cks):
              vf_c_ks = np.zeros(kiln_side_ids_ceil.size, dtype=np.float64)
              vf_c_ks[hit_cks] = (
                  cos_c_ks[hit_cks]
                  * cos_ks_c[hit_cks]
                  * a_ks[hit_cks]
                  / (math.pi * d_sq_cks[hit_cks] + 0.5 * a_ks[hit_cks])
              )
              mirror_ks_contrib = params.rho_reflector_top * vf_c_ks
              vf_vec[kiln_side_ids_ceil] += mirror_ks_contrib

        # (ii) Specular bounce from finite KILN_BOTTOM virtual image disk to ceiling
        if bot_ids.size == 1 and img_coords_b.shape[0] > 0:
          b_id = int(bot_ids[0])
          r_c_to_b = img_coords_b - pt_c[None, :]
          d_sq_cb = np.sum(r_c_to_b * r_c_to_b, axis=1)
          d_cb = np.sqrt(np.maximum(d_sq_cb, 1e-12))
          cos_c_b = -r_c_to_b[:, 2] / d_cb
          cos_img_cb = -r_c_to_b[:, 2] / d_cb
          dz_cb = pt_c[2] - params.z_mirror
          if dz_cb > 1e-6:
            t_cb = (pt_c[2] - params.z_reflector_top) / dz_cb
            x_hit_cb = pt_c[0] + t_cb * (img_coords_b[:, 0] - pt_c[0])
            y_hit_cb = pt_c[1] + t_cb * (img_coords_b[:, 1] - pt_c[1])
            r_hit_cb = np.hypot(
                x_hit_cb - params.x_kiln_center,
                y_hit_cb - params.y_kiln_center,
            )
            hit_cb = (
                (cos_c_b > 1e-6)
                & (cos_img_cb > 1e-6)
                & (r_hit_cb >= params.r_kiln - 1e-4)
                & (r_hit_cb <= params.r_reflector + 1e-4)
            )
            if np.any(hit_cb):
              hit_cb_idx = np.flatnonzero(hit_cb)
              p1_cb = np.column_stack([
                  x_hit_cb[hit_cb_idx],
                  y_hit_cb[hit_cb_idx],
                  np.full(
                      hit_cb_idx.size, params.z_reflector_top, dtype=np.float64
                  ),
              ])
              blocked_cb = _is_segment_blocked_by_solids(
                  p1=p1_cb,
                  p2=np.broadcast_to(pt_c, p1_cb.shape),
                  params=params,
                  check_kiln_cyl=True,
                  check_hood=True,
                  check_reflector_disk=False,
              )
              hit_cb[hit_cb_idx[blocked_cb]] = False
              if np.any(hit_cb):
                vf_c_b = float(
                    np.sum(
                        cos_c_b[hit_cb]
                        * cos_img_cb[hit_cb]
                        * d_area_b
                        / (math.pi * d_sq_cb[hit_cb] + 0.5 * d_area_b)
                    )
                )
                mirror_b_contrib = params.rho_reflector_top * vf_c_b
                vf_vec[b_id] += mirror_b_contrib

      vf_vec[r_id] = params.eps_reflector_top * vf_ceil_ref_direct

    s_sum = float(np.sum(vf_vec))
    if s_sum > constants.MAX_SENSOR_VF_SUM:
      vf_vec *= constants.MAX_SENSOR_VF_SUM / s_sum
    ceil_sensor_vf[name] = vf_vec

  return ceil_sensor_vf


def update_ceiling_sensor_vfs(
    vf_sys: ViewFactorSystem,
    mesh: mesh_lib.SurfaceMesh,
    params: params_lib.SimParams,
) -> ViewFactorSystem:
  """Returns a ViewFactorSystem with updated ceiling rim sensor view factors."""
  new_sensor_vf = dict(vf_sys.sensor_vf)
  new_sensor_vf.update(compute_ceiling_sensor_vfs(mesh=mesh, params=params))
  return ViewFactorSystem(
      f_eff=vf_sys.f_eff,
      f_room=vf_sys.f_room,
      sensor_vf=new_sensor_vf,
  )