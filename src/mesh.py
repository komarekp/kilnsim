"""Surface mesh discretization and single-patch inspection API.

Constructs a Structure-of-Arrays (SoA) SurfaceMesh for fast matrix operations
while providing ergonomic PatchView lookup methods by ID, SurfaceType, or 3D
spatial coordinates.
"""

import dataclasses
import enum
import math
from typing import Any, Optional
import numpy as np
from src import params as params_lib
from proto import kiln_sim_pb2


class SurfaceType(enum.IntEnum):
  """Explicitly assigned surface categories matching kiln_sim.proto."""

  UNSPECIFIED = 0
  KILN_SIDE = 1
  KILN_LID = 2
  KILN_BOTTOM = 3
  SHIELD_LEFT = 4
  SHIELD_BACK = 5
  GYPSUM_LEFT = 6
  GYPSUM_BACK = 7
  REFLECTOR_TOP = 8
  SSR_BOX = 9


@dataclasses.dataclass(frozen=True)
class PatchView:
  """Self-contained view of a single discretized surface patch for inspection.

  Bundles geometric properties, axis-aligned bounding box coordinates, chimney
  grid indices (if applicable), and solved thermal state values into a single
  record so callers do not need to index multiple parallel NumPy arrays.
  """

  patch_id: int
  surface_type: SurfaceType
  centroid_m: tuple[float, float, float]
  normal: tuple[float, float, float]
  area_m2: float
  emissivity: float
  bounds_min_m: tuple[float, float, float]
  bounds_max_m: tuple[float, float, float]
  chimney_col_row: Optional[tuple[int, int]] = None
  temp_front_c: Optional[float] = None
  temp_back_c: Optional[float] = None
  q_rad_net_w_m2: Optional[float] = None
  q_conv_net_w_m2: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class FloorSensorPoint:
  """Point sensor on the floor for evaluating subfloor and vinyl temperatures."""

  name: str
  position_m: tuple[float, float, float]
  normal: tuple[float, float, float] = (0.0, 0.0, 1.0)


class SurfaceMesh:
  """Vectorized surface patch storage with single-patch inspection helpers."""

  def __init__(
      self,
      surface_types: np.ndarray,
      centroids: np.ndarray,
      normals: np.ndarray,
      areas: np.ndarray,
      emissivities: np.ndarray,
      bounds_min: np.ndarray,
      bounds_max: np.ndarray,
      chimney_wall_id: np.ndarray,
      chimney_col_idx: np.ndarray,
      chimney_row_idx: np.ndarray,
      n_cols_left: int,
      n_cols_back: int,
      n_rows_shield: int,
      floor_sensors: dict[str, FloorSensorPoint],
  ) -> None:
    """Initializes the SurfaceMesh with parallel Structure-of-Arrays buffers."""
    self.num_patches: int = int(surface_types.shape[0])
    self.surface_types: np.ndarray = surface_types  # (N,) int32
    self.centroids: np.ndarray = centroids  # (N, 3) float64
    self.normals: np.ndarray = normals  # (N, 3) float64
    self.areas: np.ndarray = areas  # (N,) float64
    self.emissivities: np.ndarray = emissivities  # (N,) float64
    self.bounds_min: np.ndarray = bounds_min  # (N, 3) float64
    self.bounds_max: np.ndarray = bounds_max  # (N, 3) float64

    # Chimney indexing: chimney_wall_id is 0 for SHIELD_LEFT, 1 for SHIELD_BACK,
    # and -1 for non-shield patches.
    self.chimney_wall_id: np.ndarray = chimney_wall_id  # (N,) int32
    self.chimney_col_idx: np.ndarray = chimney_col_idx  # (N,) int32
    self.chimney_row_idx: np.ndarray = chimney_row_idx  # (N,) int32
    self.n_cols_left: int = n_cols_left
    self.n_cols_back: int = n_cols_back
    self.n_rows_shield: int = n_rows_shield
    self.floor_sensors: dict[str, FloorSensorPoint] = floor_sensors

  def get_patch(
      self,
      patch_id: int,
      solution: Optional[Any] = None,
  ) -> PatchView:
    """Returns a self-contained PatchView for the specified patch_id.

    Args:
      patch_id: 0-based index of the patch in [0, num_patches - 1].
      solution: Optional SolverSolution containing solved temperatures/fluxes.

    Returns:
      Populated PatchView instance.

    Raises:
      IndexError: If patch_id is out of range.
    """
    if patch_id < 0 or patch_id >= self.num_patches:
      raise IndexError(
          f"patch_id {patch_id} out of range [0, {self.num_patches - 1}]."
      )
    col_row: Optional[tuple[int, int]] = None
    if self.chimney_col_idx[patch_id] >= 0:
      col_row = (
          int(self.chimney_col_idx[patch_id]),
          int(self.chimney_row_idx[patch_id]),
      )

    t_front_c: Optional[float] = None
    t_back_c: Optional[float] = None
    q_rad: Optional[float] = None
    q_conv: Optional[float] = None
    if solution is not None:
      t_front_c = float(
          solution.t_front_k[patch_id] - params_lib.KELVIN_OFFSET
      )
      t_back_c = float(
          solution.t_back_k[patch_id] - params_lib.KELVIN_OFFSET
      )
      q_rad = float(solution.q_rad_net_w_m2[patch_id])
      q_conv = float(solution.q_conv_net_w_m2[patch_id])

    c = self.centroids[patch_id]
    n = self.normals[patch_id]
    b_min = self.bounds_min[patch_id]
    b_max = self.bounds_max[patch_id]

    return PatchView(
        patch_id=int(patch_id),
        surface_type=SurfaceType(int(self.surface_types[patch_id])),
        centroid_m=(float(c[0]), float(c[1]), float(c[2])),
        normal=(float(n[0]), float(n[1]), float(n[2])),
        area_m2=float(self.areas[patch_id]),
        emissivity=float(self.emissivities[patch_id]),
        bounds_min_m=(float(b_min[0]), float(b_min[1]), float(b_min[2])),
        bounds_max_m=(float(b_max[0]), float(b_max[1]), float(b_max[2])),
        chimney_col_row=col_row,
        temp_front_c=t_front_c,
        temp_back_c=t_back_c,
        q_rad_net_w_m2=q_rad,
        q_conv_net_w_m2=q_conv,
    )

  def get_patches_by_type(
      self,
      surface_type: SurfaceType,
      solution: Optional[Any] = None,
  ) -> list[PatchView]:
    """Returns all PatchView objects matching the given SurfaceType.

    Args:
      surface_type: SurfaceType enum value to filter by.
      solution: Optional SolverSolution containing solved temperatures/fluxes.

    Returns:
      List of matching PatchView instances in patch_id order.
    """
    matching_ids = np.flatnonzero(self.surface_types == int(surface_type))
    return [self.get_patch(int(pid), solution=solution) for pid in matching_ids]

  def find_patch_at_coord(
      self,
      x_m: float,
      y_m: float,
      z_m: float,
      surface_type: Optional[SurfaceType] = None,
      tol_m: float = 0.025,
      solution: Optional[Any] = None,
  ) -> Optional[PatchView]:
    """Finds the patch containing the given 3D point (x_m, y_m, z_m).

    Checks bounding-box containment (expanded by tol_m) and perpendicular
    plane/surface distance (`|dot(pt - centroid, normal)| <= tol_m`). If
    multiple patches lie within tol_m, returns the one whose centroid is closest
    to (x_m, y_m, z_m).

    Args:
      x_m: X coordinate in metres.
      y_m: Y coordinate in metres.
      z_m: Z coordinate in metres.
      surface_type: Optional SurfaceType filter.
      tol_m: Spatial tolerance in metres (default 25 mm).
      solution: Optional SolverSolution containing solved temperatures/fluxes.

    Returns:
      The matching PatchView, or None if no patch contains the coordinate.
    """
    pt = np.array([x_m, y_m, z_m], dtype=np.float64)
    in_bounds = np.all(
        (pt >= self.bounds_min - tol_m) & (pt <= self.bounds_max + tol_m),
        axis=1,
    )
    if surface_type is not None:
      in_bounds &= self.surface_types == int(surface_type)

    candidate_ids = np.flatnonzero(in_bounds)
    if candidate_ids.size == 0:
      return None

    diffs = pt - self.centroids[candidate_ids]
    normal_dists = np.abs(np.sum(diffs * self.normals[candidate_ids], axis=1))
    valid_mask = normal_dists <= tol_m
    valid_ids = candidate_ids[valid_mask]
    if valid_ids.size == 0:
      return None

    centroid_dists = np.linalg.norm(pt - self.centroids[valid_ids], axis=1)
    best_id = int(valid_ids[int(np.argmin(centroid_dists))])
    return self.get_patch(best_id, solution=solution)

  def to_patch_protos(
      self,
      solution: Any,
  ) -> list[kiln_sim_pb2.SimulationOutput.PatchState]:
    """Converts all mesh patches and their solved states into PatchState protos.

    Args:
      solution: Converged SolverSolution instance.

    Returns:
      List of SimulationOutput.PatchState protobuf messages.
    """
    protos: list[kiln_sim_pb2.SimulationOutput.PatchState] = []
    for pid in range(self.num_patches):
      c = self.centroids[pid]
      n = self.normals[pid]
      protos.append(
          kiln_sim_pb2.SimulationOutput.PatchState(
              patch_id=pid,
              surface_type=int(self.surface_types[pid]),
              x_m=float(c[0]),
              y_m=float(c[1]),
              z_m=float(c[2]),
              nx=float(n[0]),
              ny=float(n[1]),
              nz=float(n[2]),
              area_m2=float(self.areas[pid]),
              temp_front_c=float(
                  solution.t_front_k[pid] - params_lib.KELVIN_OFFSET
              ),
              temp_back_c=float(
                  solution.t_back_k[pid] - params_lib.KELVIN_OFFSET
              ),
              q_rad_net_w_m2=float(solution.q_rad_net_w_m2[pid]),
              q_conv_net_w_m2=float(solution.q_conv_net_w_m2[pid]),
          )
      )
    return protos


def build_mesh(params: params_lib.SimParams) -> SurfaceMesh:
  """Discretizes the simulation domain into a vectorized SurfaceMesh.

  Constructs planar rectangular patches (with exact vertex-averaged centroids)
  for shields, exposed gypsum walls, and SSR faces; cylindrical skin patches
  (with centroids evaluated directly on the curved cylinder radius) for the
  kiln jacket; and circular patches for the kiln lid, kiln base, and floor
  reflector.

  Args:
    params: Validated SimParams instance.

  Returns:
    Constructed SurfaceMesh instance.
  """
  stypes: list[int] = []
  centroids: list[tuple[float, float, float]] = []
  normals: list[tuple[float, float, float]] = []
  areas: list[float] = []
  emissivities: list[float] = []
  b_mins: list[tuple[float, float, float]] = []
  b_maxs: list[tuple[float, float, float]] = []
  c_wall_id: list[int] = []
  c_col_idx: list[int] = []
  c_row_idx: list[int] = []

  def _add_patch(
      stype: SurfaceType,
      centroid: tuple[float, float, float],
      normal: tuple[float, float, float],
      area: float,
      eps: float,
      b_min: tuple[float, float, float],
      b_max: tuple[float, float, float],
      wall_id: int = -1,
      col: int = -1,
      row: int = -1,
  ) -> None:
    stypes.append(int(stype))
    centroids.append(centroid)
    normals.append(normal)
    areas.append(area)
    emissivities.append(eps)
    b_mins.append(b_min)
    b_maxs.append(b_max)
    c_wall_id.append(wall_id)
    c_col_idx.append(col)
    c_row_idx.append(row)

  # 1. Discretize SHIELD_LEFT (along +x, starting at x = d_chimney + t_shield)
  n_cols_left = max(1, int(round(params.w_shield_left / params.dx_wall)))
  n_rows_shield = max(1, int(round(params.h_shield / params.dz_wall)))
  dx_left = params.w_shield_left / n_cols_left
  dz_shield = params.h_shield / n_rows_shield
  y_left_shield = params.y_shield_front_left_wall

  for i in range(n_cols_left):
    x0 = params.x_shield_left_start + i * dx_left
    x1 = params.x_shield_left_start + (i + 1) * dx_left
    xc = 0.5 * (x0 + x1)
    for j in range(n_rows_shield):
      z0 = params.g_floor + j * dz_shield
      z1 = params.g_floor + (j + 1) * dz_shield
      zc = 0.5 * (z0 + z1)
      _add_patch(
          stype=SurfaceType.SHIELD_LEFT,
          centroid=(xc, y_left_shield, zc),
          normal=(0.0, 1.0, 0.0),
          area=dx_left * dz_shield,
          eps=params.eps_shield,
          b_min=(x0, y_left_shield, z0),
          b_max=(x1, y_left_shield, z1),
          wall_id=0,
          col=i,
          row=j,
      )

  # 2. Discretize SHIELD_BACK (along +y, starting at y = d_chimney)
  n_cols_back = max(1, int(round(params.w_shield_back / params.dx_wall)))
  dx_back = params.w_shield_back / n_cols_back
  x_back_shield = params.x_shield_front_back_wall

  for i in range(n_cols_back):
    y0 = params.y_shield_back_start + i * dx_back
    y1 = params.y_shield_back_start + (i + 1) * dx_back
    yc = 0.5 * (y0 + y1)
    for j in range(n_rows_shield):
      z0 = params.g_floor + j * dz_shield
      z1 = params.g_floor + (j + 1) * dz_shield
      zc = 0.5 * (z0 + z1)
      _add_patch(
          stype=SurfaceType.SHIELD_BACK,
          centroid=(x_back_shield, yc, zc),
          normal=(1.0, 0.0, 0.0),
          area=dx_back * dz_shield,
          eps=params.eps_shield,
          b_min=(x_back_shield, y0, z0),
          b_max=(x_back_shield, y1, z1),
          wall_id=1,
          col=i,
          row=j,
      )

  # 3. Discretize Exposed GYPSUM_LEFT (x in [x_shield_left_end, x_max_gyp])
  gyp_band_width = 1.0
  x_gyp_start = params.x_shield_left_end
  x_gyp_end = min(params.l_room, x_gyp_start + gyp_band_width)
  w_gyp_left = max(params.dx_wall, x_gyp_end - x_gyp_start)
  n_cols_gyp_left = max(1, int(round(w_gyp_left / params.dx_wall)))
  n_rows_wall = max(1, int(round(params.h_wall / params.dz_wall)))
  dx_gyp_left = w_gyp_left / n_cols_gyp_left
  dz_wall = params.h_wall / n_rows_wall

  for i in range(n_cols_gyp_left):
    x0 = x_gyp_start + i * dx_gyp_left
    x1 = x_gyp_start + (i + 1) * dx_gyp_left
    xc = 0.5 * (x0 + x1)
    for j in range(n_rows_wall):
      z0 = j * dz_wall
      z1 = (j + 1) * dz_wall
      zc = 0.5 * (z0 + z1)
      _add_patch(
          stype=SurfaceType.GYPSUM_LEFT,
          centroid=(xc, params.t_gypsum, zc),
          normal=(0.0, 1.0, 0.0),
          area=dx_gyp_left * dz_wall,
          eps=params.eps_gypsum,
          b_min=(x0, params.t_gypsum, z0),
          b_max=(x1, params.t_gypsum, z1),
      )

  # 4. Discretize Exposed GYPSUM_BACK (y in [y_shield_back_end, y_max_gyp])
  y_gyp_start = params.y_shield_back_end
  y_gyp_end = min(params.w_room, y_gyp_start + gyp_band_width)
  w_gyp_back = max(params.dx_wall, y_gyp_end - y_gyp_start)
  n_cols_gyp_back = max(1, int(round(w_gyp_back / params.dx_wall)))
  dx_gyp_back = w_gyp_back / n_cols_gyp_back

  for i in range(n_cols_gyp_back):
    y0 = y_gyp_start + i * dx_gyp_back
    y1 = y_gyp_start + (i + 1) * dx_gyp_back
    yc = 0.5 * (y0 + y1)
    for j in range(n_rows_wall):
      z0 = j * dz_wall
      z1 = (j + 1) * dz_wall
      zc = 0.5 * (z0 + z1)
      _add_patch(
          stype=SurfaceType.GYPSUM_BACK,
          centroid=(params.t_gypsum, yc, zc),
          normal=(1.0, 0.0, 0.0),
          area=dx_gyp_back * dz_wall,
          eps=params.eps_gypsum,
          b_min=(params.t_gypsum, y0, z0),
          b_max=(params.t_gypsum, y1, z1),
      )

  # 5. Discretize KILN_SIDE Cylinder (exact on-skin cylindrical centroid)
  d_theta = (2.0 * math.pi) / params.n_theta_kiln
  dz_kiln = params.h_kiln_total / params.n_z_kiln
  patch_area_kiln = params.r_kiln * d_theta * dz_kiln

  for i in range(params.n_theta_kiln):
    theta0 = -math.pi + i * d_theta
    theta1 = -math.pi + (i + 1) * d_theta
    theta_c = 0.5 * (theta0 + theta1)
    nx = math.cos(theta_c)
    ny = math.sin(theta_c)
    xc = params.x_kiln_center + params.r_kiln * nx
    yc = params.y_kiln_center + params.r_kiln * ny

    xs = [
        params.x_kiln_center + params.r_kiln * math.cos(theta0),
        params.x_kiln_center + params.r_kiln * math.cos(theta1),
        xc,
    ]
    ys = [
        params.y_kiln_center + params.r_kiln * math.sin(theta0),
        params.y_kiln_center + params.r_kiln * math.sin(theta1),
        yc,
    ]

    for j in range(params.n_z_kiln):
      z0 = params.z_drum_bot + j * dz_kiln
      z1 = params.z_drum_bot + (j + 1) * dz_kiln
      zc = 0.5 * (z0 + z1)
      _add_patch(
          stype=SurfaceType.KILN_SIDE,
          centroid=(xc, yc, zc),
          normal=(nx, ny, 0.0),
          area=patch_area_kiln,
          eps=params.eps_kiln_side,
          b_min=(min(xs), min(ys), z0),
          b_max=(max(xs), max(ys), z1),
      )

  # 6. Single circular patch for KILN_LID top face (normal +z)
  area_kiln_disk = math.pi * (params.r_kiln ** 2)
  _add_patch(
      stype=SurfaceType.KILN_LID,
      centroid=(params.x_kiln_center, params.y_kiln_center, params.z_kiln_top),
      normal=(0.0, 0.0, 1.0),
      area=area_kiln_disk,
      eps=params.eps_kiln_top,
      b_min=(
          params.x_kiln_center - params.r_kiln,
          params.y_kiln_center - params.r_kiln,
          params.z_kiln_top,
      ),
      b_max=(
          params.x_kiln_center + params.r_kiln,
          params.y_kiln_center + params.r_kiln,
          params.z_kiln_top,
      ),
  )

  # 7. Single circular patch for KILN_BOTTOM underside face (normal -z)
  _add_patch(
      stype=SurfaceType.KILN_BOTTOM,
      centroid=(params.x_kiln_center, params.y_kiln_center, params.z_drum_bot),
      normal=(0.0, 0.0, -1.0),
      area=area_kiln_disk,
      eps=params.eps_kiln_bot,
      b_min=(
          params.x_kiln_center - params.r_kiln,
          params.y_kiln_center - params.r_kiln,
          params.z_drum_bot,
      ),
      b_max=(
          params.x_kiln_center + params.r_kiln,
          params.y_kiln_center + params.r_kiln,
          params.z_drum_bot,
      ),
  )

  # 8. Single circular patch for REFLECTOR_TOP face (normal +z)
  area_reflector = math.pi * (params.r_reflector ** 2)
  _add_patch(
      stype=SurfaceType.REFLECTOR_TOP,
      centroid=(
          params.x_kiln_center,
          params.y_kiln_center,
          params.z_reflector_top,
      ),
      normal=(0.0, 0.0, 1.0),
      area=area_reflector,
      eps=params.eps_reflector_top,
      b_min=(
          params.x_kiln_center - params.r_reflector,
          params.y_kiln_center - params.r_reflector,
          params.z_reflector_top,
      ),
      b_max=(
          params.x_kiln_center + params.r_reflector,
          params.y_kiln_center + params.r_reflector,
          params.z_reflector_top,
      ),
  )

  # 9. Discretize the 3 exposed vertical faces of the SSR_BOX
  ur_x = math.cos(params.ssr_angle_rad)
  ur_y = math.sin(params.ssr_angle_rad)
  ut_x = -ur_y
  ut_y = ur_x
  z_ssr_c = 0.5 * (params.z_ssr_bot + params.z_ssr_top)

  r_ssr_outer = params.r_kiln + params.d_ssr
  x_rear = params.x_kiln_center + r_ssr_outer * ur_x
  y_rear = params.y_kiln_center + r_ssr_outer * ur_y
  half_w = 0.5 * params.w_ssr
  dx_rear = abs(ut_x) * half_w
  dy_rear = abs(ut_y) * half_w
  _add_patch(
      stype=SurfaceType.SSR_BOX,
      centroid=(x_rear, y_rear, z_ssr_c),
      normal=(ur_x, ur_y, 0.0),
      area=params.w_ssr * params.h_ssr,
      eps=params.eps_ssr,
      b_min=(
          x_rear - dx_rear,
          y_rear - dy_rear,
          params.z_ssr_bot,
      ),
      b_max=(
          x_rear + dx_rear,
          y_rear + dy_rear,
          params.z_ssr_top,
      ),
  )

  r_ssr_mid = params.r_kiln + 0.5 * params.d_ssr
  half_d = 0.5 * params.d_ssr
  dx_side = abs(ur_x) * half_d
  dy_side = abs(ur_y) * half_d
  for sign in (-1.0, 1.0):
    x_side = (
        params.x_kiln_center
        + r_ssr_mid * ur_x
        + sign * half_w * ut_x
    )
    y_side = (
        params.y_kiln_center
        + r_ssr_mid * ur_y
        + sign * half_w * ut_y
    )
    _add_patch(
        stype=SurfaceType.SSR_BOX,
        centroid=(x_side, y_side, z_ssr_c),
        normal=(sign * ut_x, sign * ut_y, 0.0),
        area=params.d_ssr * params.h_ssr,
        eps=params.eps_ssr,
        b_min=(
            x_side - dx_side,
            y_side - dy_side,
            params.z_ssr_bot,
        ),
        b_max=(
            x_side + dx_side,
            y_side + dy_side,
            params.z_ssr_top,
        ),
    )

  # 10. Floor point sensors
  r_pad_sensor = params.r_reflector + 0.005
  diag_inv = 1.0 / math.sqrt(2.0)
  floor_sensors = {
      "pad_center": FloorSensorPoint(
          name="pad_center",
          position_m=(
              params.x_kiln_center,
              params.y_kiln_center,
              params.z_pad_top,
          ),
      ),
      "pad_edge_corner": FloorSensorPoint(
          name="pad_edge_corner",
          position_m=(
              params.x_kiln_center - r_pad_sensor * diag_inv,
              params.y_kiln_center - r_pad_sensor * diag_inv,
              params.z_pad_top,
          ),
      ),
      "pad_edge_room": FloorSensorPoint(
          name="pad_edge_room",
          position_m=(
              params.x_kiln_center + r_pad_sensor * diag_inv,
              params.y_kiln_center + r_pad_sensor * diag_inv,
              params.z_pad_top,
          ),
      ),
      "vinyl_edge_x": FloorSensorPoint(
          name="vinyl_edge_x",
          position_m=(
              params.w_floor_pad + 0.005,
              params.y_kiln_center,
              0.0,
          ),
      ),
      "vinyl_edge_y": FloorSensorPoint(
          name="vinyl_edge_y",
          position_m=(
              params.x_kiln_center,
              params.w_floor_pad + 0.005,
              0.0,
          ),
      ),
  }

  # Add radial candidate points outside the reflector along the corner and room
  # diagonals spanning across the reflector's optical shadow and penumbra up to
  # the floor pad boundary so peak floor/VIP temperatures outside the shadow
  # are captured.
  dz_gap = max(0.0, params.z_reflector_top - params.z_pad_top)
  dz_kiln_ref = max(1e-3, params.z_drum_bot - params.z_reflector_top)
  r_shadow_near = params.r_reflector + (
      max(0.0, params.r_reflector - params.r_kiln) * (dz_gap / dz_kiln_ref)
  )
  r_shadow_center = params.r_reflector * (1.0 + dz_gap / dz_kiln_ref)

  d_corner_max = min(
      params.x_kiln_center - params.x_shield_front_back_wall,
      params.y_kiln_center - params.y_shield_front_left_wall,
  )
  r_max_corner = max(r_pad_sensor, math.sqrt(2.0) * d_corner_max - 0.005)

  d_room_max = min(
      params.w_floor_pad - params.x_kiln_center,
      params.w_floor_pad - params.y_kiln_center,
  )
  r_max_room = max(r_pad_sensor, math.sqrt(2.0) * d_room_max - 0.005)

  for prefix, sign, r_max_dir in (
      ("pad_outside_corner", -1.0, r_max_corner),
      ("pad_outside_room", 1.0, r_max_room),
  ):
    candidate_radii = np.concatenate([
        np.linspace(r_pad_sensor, r_max_dir, 10),
        np.array([
            r_shadow_near + 0.005,
            r_shadow_near + 0.020,
            r_shadow_near + 0.050,
            r_shadow_center,
        ]),
    ])
    candidate_radii = np.clip(candidate_radii, r_pad_sensor, r_max_dir)
    for idx, r_val in enumerate(np.unique(np.round(candidate_radii, 5))):
      s_name = f"{prefix}_{idx}"
      floor_sensors[s_name] = FloorSensorPoint(
          name=s_name,
          position_m=(
              params.x_kiln_center + sign * float(r_val) * diag_inv,
              params.y_kiln_center + sign * float(r_val) * diag_inv,
              params.z_pad_top,
          ),
      )

  return SurfaceMesh(
      surface_types=np.array(stypes, dtype=np.int32),
      centroids=np.array(
          centroids, dtype=np.int64 if not centroids else np.float64
      ),
      normals=np.array(normals, dtype=np.float64),
      areas=np.array(areas, dtype=np.float64),
      emissivities=np.array(emissivities, dtype=np.float64),
      bounds_min=np.array(b_mins, dtype=np.float64),
      bounds_max=np.array(b_maxs, dtype=np.float64),
      chimney_wall_id=np.array(c_wall_id, dtype=np.int32),
      chimney_col_idx=np.array(c_col_idx, dtype=np.int32),
      chimney_row_idx=np.array(c_row_idx, dtype=np.int32),
      n_cols_left=n_cols_left,
      n_cols_back=n_cols_back,
      n_rows_shield=n_rows_shield,
      floor_sensors=floor_sensors,
  )