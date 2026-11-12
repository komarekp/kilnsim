"""Radiative view factor matrix and specular floor-reflector engine.

Computes the diffuse patch-to-patch view factor matrix F_direct, the specular
virtual-image matrix F_mirror reflected off the circular polished stainless
steel floor reflector, and differential point-to-patch view factors for the
floor and ceiling safety sensors.
"""

import dataclasses
import math
import numpy as np
from src import mesh as mesh_lib
from src import params as params_lib

# Spatial offsets for ceiling point sensors at z = h_wall.
CEILING_SENSOR_RIM_OFFSET_M: float = 0.010
CEILING_SENSOR_WALL_OFFSET_M: float = 0.050

# Quadrature grid resolution for integrating view factors from a floor sensor
# across the finite circular kiln bottom disk with reflector-disk occlusion.
_KILN_DISK_QUAD_R: int = 16
_KILN_DISK_QUAD_THETA: int = 48


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
    # Check rays that cross or originate on z_plane on their way up to p2.
    # Only strictly upward intersections (t > 1e-6) or rays originating inside
    # an air-less flush contact (z_reflector_bot == z_pad_top and r <= r_reflector)
    # are blocked; a ray starting outside the reflector rim on the same horizontal
    # plane (t == 0, r > r_reflector) is not blocked unless z_plane > p1_z.
    crosses = (
        (p1_arr[:, 2] < z_plane - 1e-7)
        & (p2[:, 2] > z_plane + 1e-7)
        & (dz > 1e-9)
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


def _compute_point_to_kiln_bottom_vf_with_occlusion(
    pt_coord: np.ndarray,
    pt_normal: np.ndarray,
    params: params_lib.SimParams,
) -> float:
  """Integrates the view factor from a floor point to KILN_BOTTOM with occlusion.

  Subdivides the circular kiln bottom disk (at z = z_drum_bot, radius r_kiln,
  normal (0, 0, -1)) into equal-area polar elements (n_r x n_theta) and sums
  the differential view factor over all elements whose line of sight to
  pt_coord is not obstructed by the circular reflector assembly.

  Args:
    pt_coord: (3,) coordinates of the floor sensor point in metres.
    pt_normal: (3,) unit normal vector of the floor sensor point.
    params: Validated SimParams instance.

  Returns:
    Dimensionless view factor dF_{pt -> KILN_BOTTOM} in [0.0, 1.0].
  """
  if params.r_kiln <= 0.0 or pt_coord[2] >= params.z_drum_bot - 1e-6:
    return 0.0

  # Equal-area radial rings: r_k = r_kiln * sqrt((k + 0.5) / N_r)
  k_idx = np.arange(_KILN_DISK_QUAD_R, dtype=np.float64) + 0.5
  r_vals = params.r_kiln * np.sqrt(k_idx / _KILN_DISK_QUAD_R)
  theta_vals = (
      2.0
      * math.pi
      * (np.arange(_KILN_DISK_QUAD_THETA, dtype=np.float64) + 0.5)
      / _KILN_DISK_QUAD_THETA
  )
  rr, tt = np.meshgrid(r_vals, theta_vals, indexing="ij")
  rr_flat = rr.ravel()
  tt_flat = tt.ravel()

  n_sub = rr_flat.size
  d_area = (math.pi * (params.r_kiln ** 2)) / n_sub
  sub_coords = np.column_stack([
      params.x_kiln_center + rr_flat * np.cos(tt_flat),
      params.y_kiln_center + rr_flat * np.sin(tt_flat),
      np.full(n_sub, params.z_drum_bot, dtype=np.float64),
  ])

  blocked = _is_segment_blocked_by_reflector_disk(
      p1=pt_coord, p2=sub_coords, params=params
  )
  vis_mask = ~blocked
  if not np.any(vis_mask):
    return 0.0

  r_vec = sub_coords[vis_mask] - pt_coord[None, :]
  dist_sq = np.sum(r_vec * r_vec, axis=1)
  dist = np.sqrt(np.maximum(dist_sq, 1e-12))

  cos_pt = np.sum(r_vec * pt_normal[None, :], axis=1) / dist
  # Kiln bottom normal is (0, 0, -1), so -dot(r_vec, n_bot) = r_vec[:, 2]
  cos_bot = r_vec[:, 2] / dist

  valid = (cos_pt > 1e-6) & (cos_bot > 1e-6)
  if not np.any(valid):
    return 0.0

  denom = math.pi * dist_sq[valid] + 0.5 * d_area
  vf_sub = cos_pt[valid] * cos_bot[valid] * d_area / denom
  return float(np.sum(vf_sub))


def _compute_point_to_patches_vf(
    pt_coord: np.ndarray,
    pt_normal: np.ndarray,
    patch_centroids: np.ndarray,
    patch_normals: np.ndarray,
    patch_areas: np.ndarray,
    exclude_mask: np.ndarray,
) -> np.ndarray:
  """Computes differential view factors dF_{pt -> n} from a point to patches."""
  r_vec = patch_centroids - pt_coord[None, :]
  dist_sq = np.sum(r_vec * r_vec, axis=1)
  dist = np.sqrt(np.maximum(dist_sq, 1e-12))

  cos_theta_pt = np.sum(r_vec * pt_normal[None, :], axis=1) / dist
  cos_theta_n = -np.sum(r_vec * patch_normals, axis=1) / dist

  valid = (cos_theta_pt > 1e-6) & (cos_theta_n > 1e-6) & (~exclude_mask)
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

  # 2. Symmetric exchange area matrix A_m * F_{m -> n}
  af_matrix = np.zeros((n_patches, n_patches), dtype=np.float64)
  denom_matrix = math.pi * dist_sq + 0.5 * (areas[:, None] + areas[None, :])
  af_matrix[vis] = (
      (cos_m[vis] * cos_n[vis] / denom_matrix[vis])
      * (areas[:, None] * areas[None, :])[vis]
  )

  # 3. Override coaxial disk pair (KILN_BOTTOM <-> REFLECTOR_TOP)
  bot_ids = np.flatnonzero(stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
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
    # Clear the unoccluded single-centroid approximation for non-bottom patches
    for m_idx in range(n_patches):
      if m_idx != b_id_exclusion and m_idx != r_id:
        af_matrix[m_idx, r_id] = 0.0
        af_matrix[r_id, m_idx] = 0.0

    k_ref = np.arange(_KILN_DISK_QUAD_R, dtype=np.float64) + 0.5
    r_ref_vals = params.r_reflector * np.sqrt(k_ref / _KILN_DISK_QUAD_R)
    theta_ref_vals = (
        2.0
        * math.pi
        * (np.arange(_KILN_DISK_QUAD_THETA, dtype=np.float64) + 0.5)
        / _KILN_DISK_QUAD_THETA
    )
    rr_ref, tt_ref = np.meshgrid(r_ref_vals, theta_ref_vals, indexing="ij")
    rr_ref_flat = rr_ref.ravel()
    tt_ref_flat = tt_ref.ravel()
    n_ref_sub = rr_ref_flat.size
    d_area_ref = areas[r_id] / n_ref_sub
    ref_sub_coords = np.column_stack([
        params.x_kiln_center + rr_ref_flat * np.cos(tt_ref_flat),
        params.y_kiln_center + rr_ref_flat * np.sin(tt_ref_flat),
        np.full(n_ref_sub, params.z_reflector_top, dtype=np.float64),
    ])

    cand_mask = (
        (c[:, 2] > params.z_reflector_top + 1e-4)
        & (stypes != int(mesh_lib.SurfaceType.REFLECTOR_TOP))
        & (stypes != int(mesh_lib.SurfaceType.KILN_BOTTOM))
        & (stypes != int(mesh_lib.SurfaceType.KILN_LID))
    )
    cand_ids = np.flatnonzero(cand_mask)
    if cand_ids.size > 0:
      c_cand = c[cand_ids]
      n_cand = n_vec[cand_ids]
      a_cand = areas[cand_ids]
      st_cand = stypes[cand_ids]

      r_vec_mr = ref_sub_coords[None, :, :] - c_cand[:, None, :]
      dist_sq_mr = np.sum(r_vec_mr * r_vec_mr, axis=2)
      dist_mr = np.sqrt(np.maximum(dist_sq_mr, 1e-12))
      cos_m_r = np.sum(r_vec_mr * n_cand[:, None, :], axis=2) / dist_mr
      cos_ref = -r_vec_mr[:, :, 2] / dist_mr
      vis_mr = (cos_m_r > 1e-6) & (cos_ref > 1e-6)
      annulus_only_rows = (
          (st_cand == int(mesh_lib.SurfaceType.KILN_SIDE))
          | (st_cand == int(mesh_lib.SurfaceType.SSR_BOX))
      )
      if np.any(annulus_only_rows):
        vis_mr[annulus_only_rows, :] &= (
            rr_ref_flat[None, :] >= params.r_kiln - 1e-4
        )
      vis_m_idx, vis_q_idx = np.nonzero(vis_mr)
      if vis_m_idx.size > 0:
        blocked_cyl = _is_segment_blocked_by_kiln_cylinder(
            p1=ref_sub_coords[vis_q_idx],
            p2=c_cand[vis_m_idx],
            params=params,
        )
        vis_mr[vis_m_idx[blocked_cyl], vis_q_idx[blocked_cyl]] = False
      denom_mr = math.pi * dist_sq_mr + 0.5 * (a_cand[:, None] + d_area_ref)
      af_sub = np.where(
          vis_mr,
          cos_m_r * cos_ref * a_cand[:, None] * d_area_ref / denom_mr,
          0.0,
      )
      af_cand_ref = np.sum(af_sub, axis=1)
      af_matrix[cand_ids, r_id] = af_cand_ref
      af_matrix[r_id, cand_ids] = af_cand_ref

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
        & (stypes != int(mesh_lib.SurfaceType.KILN_SIDE))
    )
    target_ids = np.flatnonzero(target_mask)

    # 4a. Specular reflection of the finite KILN_BOTTOM disk (at z = z_mirror)
    if bot_ids.size == 1 and target_ids.size > 0:
      b_id = int(bot_ids[0])
      k_idx = np.arange(_KILN_DISK_QUAD_R, dtype=np.float64) + 0.5
      r_vals = params.r_kiln * np.sqrt(k_idx / _KILN_DISK_QUAD_R)
      theta_vals = (
          2.0
          * math.pi
          * (np.arange(_KILN_DISK_QUAD_THETA, dtype=np.float64) + 0.5)
          / _KILN_DISK_QUAD_THETA
      )
      rr_b, tt_b = np.meshgrid(r_vals, theta_vals, indexing="ij")
      rr_b_flat = rr_b.ravel()
      tt_b_flat = tt_b.ravel()
      n_sub_b = rr_b_flat.size
      d_area_b = areas[b_id] / n_sub_b
      img_coords_b = np.column_stack([
          params.x_kiln_center + rr_b_flat * np.cos(tt_b_flat),
          params.y_kiln_center + rr_b_flat * np.sin(tt_b_flat),
          np.full(n_sub_b, params.z_mirror, dtype=np.float64),
      ])

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
      t_hit_b = (c_t[:, 2] - params.z_reflector_top) / np.maximum(dz_tb, 1e-6)
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
      # SSR_BOX sits outside the kiln footprint and only sees reflections off
      # the overhanging annulus (r_hit >= r_kiln).
      ssr_target_rows = st_t == int(mesh_lib.SurfaceType.SSR_BOX)
      if np.any(ssr_target_rows):
        hit_b[ssr_target_rows, :] &= (
            r_hit_b[ssr_target_rows, :] >= params.r_kiln - 1e-4
        )

      # Block any reflected ray that bounces on the far side of the kiln and
      # would have to pass through the solid kiln drum to reach the target patch.
      hit_m_idx, hit_q_idx = np.nonzero(hit_b)
      if hit_m_idx.size > 0:
        p1_hits = np.column_stack([
            x_hit_b[hit_m_idx, hit_q_idx],
            y_hit_b[hit_m_idx, hit_q_idx],
            np.full(hit_m_idx.size, params.z_reflector_top, dtype=np.float64),
        ])
        p2_targets = c_t[hit_m_idx]
        # Check kiln-cylinder occlusion for bounces outside the kiln footprint
        # or on the opposite side of the kiln center from the target patch.
        dot_side = (
            (p1_hits[:, 0] - params.x_kiln_center)
            * (p2_targets[:, 0] - params.x_kiln_center)
            + (p1_hits[:, 1] - params.y_kiln_center)
            * (p2_targets[:, 1] - params.y_kiln_center)
        )
        check_cyl_mask = (
            r_hit_b[hit_m_idx, hit_q_idx] >= params.r_kiln - 1e-4
        ) | (dot_side < 0.0)
        if np.any(check_cyl_mask):
          chk_indices = np.flatnonzero(check_cyl_mask)
          blocked_b = _is_segment_blocked_by_kiln_cylinder(
              p1=p1_hits[chk_indices],
              p2=p2_targets[chk_indices],
              params=params,
          )
          bad_indices = chk_indices[blocked_b]
          hit_b[hit_m_idx[bad_indices], hit_q_idx[bad_indices]] = False

      denom_tb = math.pi * d_sq_tb + 0.5 * (a_t[:, None] + d_area_b)
      vf_tb_sub = np.where(
          hit_b, cos_t * cos_img_b * d_area_b / denom_tb, 0.0
      )
      vf_t_to_b = np.minimum(np.sum(vf_tb_sub, axis=1), 0.35)

      f_mirror[target_ids, b_id] += params.rho_reflector_top * vf_t_to_b
      f_mirror[b_id, target_ids] += (
          params.rho_reflector_top * vf_t_to_b * a_t / areas[b_id]
      )

    # 4b. Specular reflection of KILN_SIDE patches off the overhanging
    # reflector annulus (r in [r_kiln, r_reflector]) onto walls and SSR_BOX.
    kiln_side_ids = np.flatnonzero(
        stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
    )
    if (
        kiln_side_ids.size > 0
        and target_ids.size > 0
        and params.r_reflector > params.r_kiln + 1e-4
    ):
      c_ks = c[kiln_side_ids]
      n_ks = n_vec[kiln_side_ids]
      a_ks = areas[kiln_side_ids]

      c_ks_img = c_ks.copy()
      c_ks_img[:, 2] = 2.0 * params.z_reflector_top - c_ks[:, 2]
      n_ks_img = n_ks.copy()
      n_ks_img[:, 2] = -n_ks[:, 2]

      c_t = c[target_ids]
      n_t = n_vec[target_ids]
      a_t = areas[target_ids]

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
        blocked_ks = _is_segment_blocked_by_kiln_cylinder(
            p1=p1_ks_hits, p2=p2_ks_targets, params=params
        )
        hit_ks[ks_m_idx[blocked_ks], ks_k_idx[blocked_ks]] = False

      denom_tks = math.pi * d_sq_tks + 0.5 * (a_t[:, None] + a_ks[None, :])
      vf_t_to_ks = np.where(
          hit_ks, cos_t_ks * cos_ks * a_ks[None, :] / denom_tks, 0.0
      )
      f_mirror[np.ix_(target_ids, kiln_side_ids)] += (
          params.rho_reflector_top * vf_t_to_ks
      )
      f_mirror[np.ix_(kiln_side_ids, target_ids)] += (
          params.rho_reflector_top
          * vf_t_to_ks.T
          * (a_t[None, :] / a_ks[:, None])
      )

  # Deduct specularly redirected rays from the direct view factor to
  # REFLECTOR_TOP (retaining at least the absorbed fraction eps_reflector_top)
  # so that reflected photons are not double-counted in row sums.
  if ref_ids.size == 1 and params.r_reflector > 0.0:
    r_id = int(ref_ids[0])
    mirror_row_sum = np.sum(f_mirror, axis=1)
    f_direct_to_ref_rem = np.maximum(
        params.eps_reflector_top * f_direct[:, r_id],
        f_direct[:, r_id] - mirror_row_sum,
    )
    f_direct[:, r_id] = f_direct_to_ref_rem
    f_direct[r_id, :] = f_direct_to_ref_rem * areas / areas[r_id]

  f_eff = f_direct + f_mirror

  # 5. Enforce row sum <= 0.995
  row_sums = np.sum(f_eff, axis=1)
  over_mask = row_sums > 0.995
  if np.any(over_mask):
    scale = 0.995 / row_sums[over_mask]
    f_eff[over_mask, :] *= scale[:, None]
    row_sums = np.sum(f_eff, axis=1)

  f_room = np.maximum(0.0, 1.0 - row_sums)

  # 6. Floor point-to-patch view factor vectors (accounting for occlusion by
  # the circular reflector disk and finite-disk integration over KILN_BOTTOM).
  sensor_vf: dict[str, np.ndarray] = {}
  exclude_floor = stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  for name, sensor in mesh.floor_sensors.items():
    pt_c = np.array(sensor.position_m, dtype=np.float64)
    pt_n = np.array(sensor.normal, dtype=np.float64)
    blocked_patches = _is_segment_blocked_by_reflector_disk(
        p1=pt_c, p2=c, params=params
    )
    vf_vec = _compute_point_to_patches_vf(
        pt_coord=pt_c,
        pt_normal=pt_n,
        patch_centroids=c,
        patch_normals=n_vec,
        patch_areas=areas,
        exclude_mask=exclude_floor | blocked_patches,
    )
    if bot_ids.size == 1:
      vf_vec[int(bot_ids[0])] = _compute_point_to_kiln_bottom_vf_with_occlusion(
          pt_coord=pt_c,
          pt_normal=pt_n,
          params=params,
      )
    s_sum = float(np.sum(vf_vec))
    if s_sum > 0.99:
      vf_vec *= 0.99 / s_sum
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
  exclude_ceil = (stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM)) | (
      stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  )
  kiln_side_ids_ceil = np.flatnonzero(
      stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
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
    )
    if ref_ids.size == 1 and params.r_reflector > params.r_kiln + 1e-4:
      r_id = int(ref_ids[0])
      r_inner = params.r_kiln
      r_outer = params.r_reflector
      k_ann = np.arange(_KILN_DISK_QUAD_R, dtype=np.float64) + 0.5
      r_ann = np.sqrt(
          r_inner * r_inner
          + (r_outer * r_outer - r_inner * r_inner)
          * (k_ann / _KILN_DISK_QUAD_R)
      )
      theta_ann = (
          2.0
          * math.pi
          * (np.arange(_KILN_DISK_QUAD_THETA, dtype=np.float64) + 0.5)
          / _KILN_DISK_QUAD_THETA
      )
      rr_a, tt_a = np.meshgrid(r_ann, theta_ann, indexing="ij")
      rr_a_flat = rr_a.ravel()
      tt_a_flat = tt_a.ravel()
      n_ann = rr_a_flat.size
      d_area_ann = (
          math.pi * (r_outer * r_outer - r_inner * r_inner)
      ) / n_ann
      ann_coords = np.column_stack([
          params.x_kiln_center + rr_a_flat * np.cos(tt_a_flat),
          params.y_kiln_center + rr_a_flat * np.sin(tt_a_flat),
          np.full(n_ann, params.z_reflector_top, dtype=np.float64),
      ])
      blocked_ann = _is_segment_blocked_by_kiln_cylinder(
          p1=ann_coords, p2=np.broadcast_to(pt_c, ann_coords.shape), params=params
      )
      vis_ann = ~blocked_ann
      vf_ceil_ref_direct = 0.0
      if np.any(vis_ann):
        r_vec_ca = ann_coords[vis_ann] - pt_c[None, :]
        dist_sq_ca = np.sum(r_vec_ca * r_vec_ca, axis=1)
        dist_ca = np.sqrt(np.maximum(dist_sq_ca, 1e-12))
        cos_ceil = -r_vec_ca[:, 2] / dist_ca
        cos_ref_a = -r_vec_ca[:, 2] / dist_ca
        valid_ca = (cos_ceil > 1e-6) & (cos_ref_a > 1e-6)
        if np.any(valid_ca):
          denom_ca = (
              math.pi * dist_sq_ca[valid_ca] + 0.5 * d_area_ann
          )
          vf_ceil_ref_direct = float(
              np.sum(
                  cos_ceil[valid_ca]
                  * cos_ref_a[valid_ca]
                  * d_area_ann
                  / denom_ca
              )
          )

      vf_ceil_mirror_sum = 0.0
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
            blocked_cks = _is_segment_blocked_by_kiln_cylinder(
                p1=p1_cks,
                p2=np.broadcast_to(pt_c, p1_cks.shape),
                params=params,
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
              vf_ceil_mirror_sum += float(np.sum(mirror_ks_contrib))

        # (ii) Specular bounce from finite KILN_BOTTOM virtual image disk to ceiling
        if bot_ids.size == 1:
          b_id = int(bot_ids[0])
          k_idx = np.arange(_KILN_DISK_QUAD_R, dtype=np.float64) + 0.5
          r_vals = params.r_kiln * np.sqrt(k_idx / _KILN_DISK_QUAD_R)
          theta_vals = (
              2.0
              * math.pi
              * (np.arange(_KILN_DISK_QUAD_THETA, dtype=np.float64) + 0.5)
              / _KILN_DISK_QUAD_THETA
          )
          rr_b, tt_b = np.meshgrid(r_vals, theta_vals, indexing="ij")
          rr_b_flat = rr_b.ravel()
          tt_b_flat = tt_b.ravel()
          n_sub_b = rr_b_flat.size
          d_area_b = areas[b_id] / n_sub_b
          img_coords_b = np.column_stack([
              params.x_kiln_center + rr_b_flat * np.cos(tt_b_flat),
              params.y_kiln_center + rr_b_flat * np.sin(tt_b_flat),
              np.full(n_sub_b, params.z_mirror, dtype=np.float64),
          ])
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
              blocked_cb = _is_segment_blocked_by_kiln_cylinder(
                  p1=p1_cb,
                  p2=np.broadcast_to(pt_c, p1_cb.shape),
                  params=params,
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
                vf_ceil_mirror_sum += mirror_b_contrib

      vf_vec[r_id] = max(
          params.eps_reflector_top * vf_ceil_ref_direct,
          vf_ceil_ref_direct - vf_ceil_mirror_sum,
      )

    s_sum = float(np.sum(vf_vec))
    if s_sum > 0.99:
      vf_vec *= 0.99 / s_sum
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