"""Safety metrics evaluation, point-temperature queries, and output assembly."""

import math
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from src import constants
from src import mesh as mesh_lib
from src import params as params_lib
from src import solver as solver_lib
from src import view_factors as vf_lib
from proto import kiln_sim_pb2

# Engineering pass/fail temperature thresholds in degrees Celsius (re-exported).
LIMIT_GYPSUM_C: float = constants.LIMIT_GYPSUM_C
LIMIT_CEILING_C: float = constants.LIMIT_CEILING_C
LIMIT_SSR_AIR_C: float = constants.LIMIT_SSR_AIR_C
LIMIT_SSR_CASE_C: float = constants.LIMIT_SSR_CASE_C
LIMIT_VIP_C: float = constants.LIMIT_VIP_C
LIMIT_VINYL_C: float = constants.LIMIT_VINYL_C
LIMIT_SHIELD_C: float = constants.LIMIT_SHIELD_C
LIMIT_AAC_C: float = constants.LIMIT_AAC_C

# Floor surface emissivities and under-reflector / ceiling mixing coefficients.
FLOOR_PAD_EMISSIVITY: float = constants.FLOOR_PAD_EMISSIVITY
VINYL_FLOOR_EMISSIVITY: float = constants.VINYL_FLOOR_EMISSIVITY
H_CONV_UNDER_REFLECTOR_W_M2K: float = constants.H_CONV_UNDER_REFLECTOR_W_M2K
CHIMNEY_CEILING_MIXING_FRACTION: float = constants.CHIMNEY_CEILING_MIXING_FRACTION


def get_kiln_front_face_temp_c(
    mesh: mesh_lib.SurfaceMesh,
    solution: solver_lib.SolverSolution,
    params: params_lib.SimParams,
    angle_deg: float = 0.0,
) -> float:
  """Returns the equilibrium surface temperature of the front face of the kiln."""
  angle_rad = math.radians(angle_deg)
  x_m = params.x_kiln_center + params.r_kiln * math.cos(angle_rad)
  y_m = params.y_kiln_center + params.r_kiln * math.sin(angle_rad)
  z_m = params.z_drum_bot + 0.5 * params.h_drum

  patch = mesh.find_patch_at_coord(
      x_m=x_m,
      y_m=y_m,
      z_m=z_m,
      surface_type=mesh_lib.SurfaceType.KILN_SIDE,
      tol_m=max(0.05, params.h_kiln_total / params.n_z_kiln),
      solution=solution,
  )
  if patch is None or patch.temp_front_c is None:
    raise ValueError(
        f"No KILN_SIDE patch found at front face ({x_m:.3f}, {y_m:.3f}, "
        f"{z_m:.3f}) m."
    )
  return patch.temp_front_c


def get_shield_temp_behind_kiln_c(
    mesh: mesh_lib.SurfaceMesh,
    solution: solver_lib.SolverSolution,
    params: params_lib.SimParams,
) -> float:
  """Returns the calcium silicate shield temperature on the +x axis by the kiln."""
  x_m = params.x_kiln_center
  y_m = params.y_shield_front_left_wall
  z_m = params.z_drum_bot + 0.5 * params.h_drum

  patch = mesh.find_patch_at_coord(
      x_m=x_m,
      y_m=y_m,
      z_m=z_m,
      surface_type=mesh_lib.SurfaceType.SHIELD_LEFT,
      tol_m=max(0.05, params.dx_wall, params.dz_wall),
      solution=solution,
  )
  if patch is None or patch.temp_front_c is None:
    raise ValueError(
        f"No SHIELD_LEFT patch found at ({x_m:.3f}, {y_m:.3f}, {z_m:.3f}) m."
    )
  return patch.temp_front_c


def _solve_point_sensor_balance_k(
    vf_vec: np.ndarray,
    radiosity_w_m2: np.ndarray,
    eps_surface: float,
    u_backing: float,
    t_amb_k: float,
    t_ext_k: float,
    h_conv_fn,
) -> float:
  """Solves radiative-convective-conductive equilibrium at a point sensor."""
  f_room = max(0.0, 1.0 - float(np.sum(vf_vec)))
  g_incident = float(np.dot(vf_vec, radiosity_w_m2)) + (
      f_room * params_lib.STEFAN_BOLTZMANN * (t_amb_k ** 4)
  )
  q_abs = eps_surface * g_incident

  def _res(t_k: float) -> float:
    q_emit = eps_surface * constants.STEFAN_BOLTZMANN * (t_k ** 4)
    dt = t_k - t_amb_k
    q_conv = h_conv_fn(dt) * dt
    q_cond = u_backing * (t_k - t_ext_k)
    return q_abs - q_emit - q_conv - q_cond

  sol = optimize.root_scalar(
      _res,
      bracket=[min(t_amb_k, t_ext_k) - 50.0, max(t_amb_k, t_ext_k) + 500.0],
      method="brentq",
  )
  return float(sol.root)


def _solve_horizontal_sensor_temp_k(
    vf_vec: np.ndarray,
    radiosity_w_m2: np.ndarray,
    eps_surface: float,
    u_backing: float,
    t_amb_k: float,
    t_ext_k: float,
    surface_faces_up: bool,
) -> float:
  """Solves a horizontal floor/ceiling sensor using stability-aware h_c."""

  def _h_horiz(dt: float) -> float:
    is_unstable = (dt > 0.0) if surface_faces_up else (dt < 0.0)
    if is_unstable:
      return max(
          constants.H_CONV_MIN_W_M2K,
          constants.H_CONV_HORIZ_UNSTABLE_COEFF * (abs(dt) ** 0.25),
      )
    return max(
        constants.H_CONV_STABLE_MIN_W_M2K,
        constants.H_CONV_HORIZ_STABLE_COEFF * (abs(dt) ** 0.20),
    )

  return _solve_point_sensor_balance_k(
      vf_vec=vf_vec,
      radiosity_w_m2=radiosity_w_m2,
      eps_surface=eps_surface,
      u_backing=u_backing,
      t_amb_k=t_amb_k,
      t_ext_k=t_ext_k,
      h_conv_fn=_h_horiz,
  )


def _solve_floor_sensor_temp_k(
    vf_vec: np.ndarray,
    radiosity_w_m2: np.ndarray,
    eps_floor: float,
    u_subfloor: float,
    t_amb_k: float,
    t_ext_k: float,
) -> float:
  """Solves the local radiative-convective-conductive equilibrium on the floor."""
  return _solve_horizontal_sensor_temp_k(
      vf_vec=vf_vec,
      radiosity_w_m2=radiosity_w_m2,
      eps_surface=eps_floor,
      u_backing=u_subfloor,
      t_amb_k=t_amb_k,
      t_ext_k=t_ext_k,
      surface_faces_up=True,
  )


def _solve_floor_2d_field(
    mesh: mesh_lib.SurfaceMesh,
    vf_sys: vf_lib.ViewFactorSystem,
    solution: solver_lib.SolverSolution,
    params: params_lib.SimParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Solves 2D lateral pad+screed conduction coupled to VIP and 4 stand legs."""
  xs = vf_sys.floor_grid_x_m
  ys = vf_sys.floor_grid_y_m
  vf_grid = vf_sys.floor_grid_vf
  if xs is None or ys is None or vf_grid is None:
    raise ValueError("ViewFactorSystem does not contain 2D floor grid VFs.")

  nx = int(xs.size)
  ny = int(ys.size)
  n_cells = nx * ny
  dx = float(xs[1] - xs[0]) if nx > 1 else constants.FLOOR_2D_CELL_SIZE_M
  dy = float(ys[1] - ys[0]) if ny > 1 else constants.FLOOR_2D_CELL_SIZE_M
  d_area = dx * dy

  xx, yy = np.meshgrid(xs, ys, indexing="ij")
  gx = xx.ravel()
  gy = yy.ravel()
  in_pad_2d = (xx <= params.w_floor_pad + 1e-9) & (
      yy <= params.w_floor_pad + 1e-9
  )
  in_pad = in_pad_2d.ravel()
  r_cell = np.hypot(gx - params.x_kiln_center, gy - params.y_kiln_center)
  under_ref = (
      (r_cell <= params.r_reflector)
      if params.r_reflector > 0.0
      else np.zeros(n_cells, dtype=bool)
  )

  r_pad = (
      (params.t_floor_pad / max(params.k_pad, 0.05))
      if params.t_floor_pad > 0.0
      else 0.0
  )
  r_screed = params.t_screed / max(params.k_screed, 0.05)
  r_vip = params.t_vip / max(params.k_vip, 1e-4)
  u_12 = np.where(
      in_pad,
      1.0 / max(r_pad + 0.5 * r_screed, 1e-4),
      1.0 / max(0.5 * r_screed, 1e-4),
  )
  u_2_ext = 1.0 / max(0.5 * r_screed + r_vip, 1e-4)

  # Assemble fixed 2N x 2N graph-Laplacian + vertical + ground conductances.
  rows_list: list[np.ndarray] = []
  cols_list: list[np.ndarray] = []
  vals_list: list[np.ndarray] = []
  diag_fixed = np.zeros(2 * n_cells, dtype=np.float64)

  def _add_edges(u_idx: np.ndarray, v_idx: np.ndarray, g: np.ndarray) -> None:
    pos = g > 0.0
    if not np.any(pos):
      return
    u = u_idx[pos]
    v = v_idx[pos]
    gp = g[pos]
    rows_list.extend([u, v])
    cols_list.extend([v, u])
    vals_list.extend([-gp, -gp])
    np.add.at(diag_fixed, u, gp)
    np.add.at(diag_fixed, v, gp)

  k_lat_pad = params.k_pad * max(0.0, params.t_floor_pad)
  k_lat_screed = params.k_screed * params.t_screed
  grid_ids = np.arange(n_cells, dtype=np.int32).reshape(nx, ny)

  if nx > 1:
    u_x = grid_ids[:-1, :].ravel()
    v_x = grid_ids[1:, :].ravel()
    pad_x = (in_pad_2d[:-1, :] & in_pad_2d[1:, :]).ravel()
    gx_top = np.where(pad_x, k_lat_pad * (dy / dx), 0.0)
    gx_scr = np.full(u_x.size, k_lat_screed * (dy / dx), dtype=np.float64)
    _add_edges(u_x, v_x, gx_top)
    _add_edges(u_x + n_cells, v_x + n_cells, gx_scr)

  if ny > 1:
    u_y = grid_ids[:, :-1].ravel()
    v_y = grid_ids[:, 1:].ravel()
    pad_y = (in_pad_2d[:, :-1] & in_pad_2d[:, 1:]).ravel()
    gy_top = np.where(pad_y, k_lat_pad * (dx / dy), 0.0)
    gy_scr = np.full(u_y.size, k_lat_screed * (dx / dy), dtype=np.float64)
    _add_edges(u_y, v_y, gy_top)
    _add_edges(u_y + n_cells, v_y + n_cells, gy_scr)

  top_nodes = np.arange(n_cells, dtype=np.int32)
  scr_nodes = top_nodes + n_cells
  g_12 = u_12 * d_area
  _add_edges(top_nodes, scr_nodes, g_12)

  g_2_ext = u_2_ext * d_area
  diag_fixed[scr_nodes] += g_2_ext

  b_fixed = np.zeros(2 * n_cells, dtype=np.float64)
  b_fixed[scr_nodes] += g_2_ext * params.t_ext_k

  # 4 steel kiln stand leg 1D fin bridges conducted to the subfloor surface.
  stypes = mesh.surface_types
  t_amb_k = float(solution.room.t_room_lower_k)
  side_mask = stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
  bot_mask = stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM)
  if params.r_kiln > 0.0 and params.h_kiln_base > 0.0 and np.any(side_mask):
    z_side = mesh.centroids[side_mask, 2]
    z_min_side = float(np.min(z_side))
    low_side = side_mask & (mesh.centroids[:, 2] <= z_min_side + 1e-4)
    t_jacket_low_k = float(
        np.sum(solution.t_front_k[low_side] * mesh.areas[low_side])
        / np.sum(mesh.areas[low_side])
    )
    t_bot_k = (
        float(solution.t_front_k[bot_mask][0])
        if np.any(bot_mask)
        else t_jacket_low_k
    )
    w_j = constants.KILN_LEG_JACKET_TEMP_WEIGHT
    t_leg_top_k = w_j * t_jacket_low_k + (1.0 - w_j) * t_bot_k

    l_leg = max(0.01, params.z_drum_bot - params.z_pad_top)
    k_a = constants.KILN_LEG_K_W_MK * constants.KILN_LEG_METAL_AREA_M2
    h_p = constants.KILN_LEG_H_EFF_W_M2K * constants.KILN_LEG_PERIMETER_M
    m_fin = math.sqrt(h_p / max(k_a, 1e-12))
    g0 = math.sqrt(h_p * k_a)
    ml = min(25.0, m_fin * l_leg)
    g_self = g0 / max(math.tanh(ml), 1e-9)
    t_th_k = t_amb_k + (t_leg_top_k - t_amb_k) / math.cosh(ml)
    g_leg_eff = 1.0 / (
        (1.0 / max(g_self, 1e-9)) + constants.KILN_LEG_FOOT_CONTACT_R_K_W
    )

    r_leg = constants.KILN_LEG_RADIUS_FRACTION * params.r_kiln
    leg_angles = (
        0.25 * math.pi,
        0.75 * math.pi,
        -0.75 * math.pi,
        -0.25 * math.pi,
    )[: constants.NUM_KILN_LEGS]
    for phi in leg_angles:
      x_leg = params.x_kiln_center + r_leg * math.cos(phi)
      y_leg = params.y_kiln_center + r_leg * math.sin(phi)
      ix = int(np.argmin(np.abs(xs - x_leg)))
      iy = int(np.argmin(np.abs(ys - y_leg)))
      leg_cell = ix * ny + iy
      diag_fixed[leg_cell] += g_leg_eff
      b_fixed[leg_cell] += g_leg_eff * t_th_k

  all_idx = np.arange(2 * n_cells, dtype=np.int32)
  rows_list.append(all_idx)
  cols_list.append(all_idx)
  vals_list.append(diag_fixed)
  l_fixed = sparse.csc_matrix(
      (
          np.concatenate(vals_list),
          (np.concatenate(rows_list), np.concatenate(cols_list)),
      ),
      shape=(2 * n_cells, 2 * n_cells),
  )

  # Precompute exposed irradiance & under-reflector thermal boundary parameters.
  sigma = constants.STEFAN_BOLTZMANN
  eps_cell = np.where(in_pad, FLOOR_PAD_EMISSIVITY, VINYL_FLOOR_EMISSIVITY)
  vf_sum = np.sum(vf_grid, axis=1)
  f_room_cell = np.maximum(0.0, 1.0 - vf_sum)
  g_inc = vf_grid @ solution.radiosity_w_m2 + (
      f_room_cell * sigma * (t_amb_k ** 4)
  )
  q_abs_exp = eps_cell * g_inc

  ref_mask = stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  has_ref = params.r_reflector > 0.0 and np.any(ref_mask) and np.any(under_ref)
  is_flush_ref = params.g_floor_reflector <= constants.GEOM_TOL_M
  has_calsil = params.t_reflector_calsil > constants.GEOM_TOL_M
  r_calsil = (
      (params.t_reflector_calsil / max(params.k_shield, 0.05))
      if has_calsil
      else 0.0
  )
  t_ref_front_k = float(solution.t_front_k[ref_mask][0]) if has_ref else t_amb_k
  t_ref_back_k = t_ref_front_k
  if has_ref:
    _, _, t_ref_back_k, _ = solver_lib.solve_under_reflector_column(
        t_ref_front_k=t_ref_front_k,
        t_room_lower_k=t_amb_k,
        params=params,
    )
  eps_underside = params.eps_shield if has_calsil else params.eps_reflector_top
  eps_gap = 1.0 / (
      (1.0 / max(eps_underside, 1e-3))
      + (1.0 / FLOOR_PAD_EMISSIVITY)
      - 1.0
  )
  g_flush = 1.0 / max(r_calsil, 1e-3)

  x = np.full(2 * n_cells, 0.5 * (t_amb_k + params.t_ext_k), dtype=np.float64)
  zero_pad = np.zeros(n_cells, dtype=np.float64)
  for _ in range(constants.FLOOR_2D_NEWTON_ITERS):
    t_surf = x[:n_cells]
    dt = t_surf - t_amb_k
    h_exp = np.where(
        dt > 0.0,
        np.maximum(
            constants.H_CONV_MIN_W_M2K,
            constants.H_CONV_HORIZ_UNSTABLE_COEFF * (np.abs(dt) ** 0.25),
        ),
        np.maximum(
            constants.H_CONV_STABLE_MIN_W_M2K,
            constants.H_CONV_HORIZ_STABLE_COEFF * (np.abs(dt) ** 0.20),
        ),
    )
    q_top_exp = (
        q_abs_exp - eps_cell * sigma * (t_surf ** 4) - h_exp * dt
    ) * d_area
    s_top_exp = (4.0 * eps_cell * sigma * (t_surf ** 3) + h_exp) * d_area

    if has_ref:
      if is_flush_ref:
        q_top_ref = g_flush * (t_ref_front_k - t_surf) * d_area
        s_top_ref = np.full(n_cells, g_flush * d_area, dtype=np.float64)
      else:
        q_top_ref = (
            eps_gap * sigma * ((t_ref_back_k ** 4) - (t_surf ** 4))
            + constants.H_CONV_UNDER_REFLECTOR_W_M2K * (t_amb_k - t_surf)
        ) * d_area
        s_top_ref = (
            4.0 * eps_gap * sigma * (t_surf ** 3)
            + constants.H_CONV_UNDER_REFLECTOR_W_M2K
        ) * d_area
      q_top = np.where(under_ref, q_top_ref, q_top_exp)
      s_top = np.where(under_ref, s_top_ref, s_top_exp)
    else:
      q_top = q_top_exp
      s_top = s_top_exp

    j_mat = l_fixed + sparse.diags(
        np.concatenate([s_top, zero_pad]), format="csc"
    )
    rhs = b_fixed.copy()
    rhs[:n_cells] += q_top + s_top * t_surf
    x = sparse_linalg.spsolve(j_mat, rhs)

  t_surf_2d_k = x[:n_cells].reshape(nx, ny)
  t_screed_2d_k = x[n_cells:].reshape(nx, ny)
  t_vip_2d_k = params.t_ext_k + (
      (t_screed_2d_k - params.t_ext_k) * (r_vip / (0.5 * r_screed + r_vip))
  )
  under_ref_2d = under_ref.reshape(nx, ny)
  return t_surf_2d_k, t_vip_2d_k, in_pad_2d, under_ref_2d


def _solve_ceiling_sensors_vec_k(
    vf_mat: np.ndarray,
    radiosity_w_m2: np.ndarray,
    eps_ceil: float,
    u_roof: float,
    t_amb_vec_k: np.ndarray,
    t_ext_k: float,
) -> np.ndarray:
  """Solves downward-facing ceiling sensor balances vectorized across a grid."""
  f_room = np.maximum(0.0, 1.0 - np.sum(vf_mat, axis=1))
  sigma = constants.STEFAN_BOLTZMANN
  g_incident = (vf_mat @ radiosity_w_m2) + f_room * sigma * (t_amb_vec_k ** 4)
  q_abs = eps_ceil * g_incident
  t_k = np.maximum(t_amb_vec_k, t_ext_k) + 2.0

  for _ in range(12):
    dt = t_k - t_amb_vec_k
    abs_dt = np.maximum(np.abs(dt), 1e-12)
    is_unstable = dt < 0.0
    h_unstable = np.maximum(
        constants.H_CONV_MIN_W_M2K,
        constants.H_CONV_HORIZ_UNSTABLE_COEFF * (abs_dt ** 0.25),
    )
    h_stable = np.maximum(
        constants.H_CONV_STABLE_MIN_W_M2K,
        constants.H_CONV_HORIZ_STABLE_COEFF * (abs_dt ** 0.20),
    )
    h_c = np.where(is_unstable, h_unstable, h_stable)
    dh_slope = np.where(is_unstable, 1.25, 1.20)
    res = (
        q_abs
        - eps_ceil * sigma * (t_k ** 4)
        - h_c * dt
        - u_roof * (t_k - t_ext_k)
    )
    jac = 4.0 * eps_ceil * sigma * (t_k ** 3) + dh_slope * h_c + u_roof
    step = np.clip(res / np.maximum(jac, 1e-6), -25.0, 25.0)
    t_k = np.maximum(constants.SOLVER_MIN_TEMP_K, t_k + step)
    if float(np.max(np.abs(res))) < 1e-6:
      break
  return t_k


def _solve_ceiling_sensor_temp_k(
    vf_vec: np.ndarray,
    radiosity_w_m2: np.ndarray,
    eps_ceil: float,
    u_roof: float,
    t_amb_k: float,
    t_ext_k: float,
) -> float:
  """Solves the downward-facing ceiling sensor balance at z = h_wall."""
  return _solve_horizontal_sensor_temp_k(
      vf_vec=vf_vec,
      radiosity_w_m2=radiosity_w_m2,
      eps_surface=eps_ceil,
      u_backing=u_roof,
      t_amb_k=t_amb_k,
      t_ext_k=t_ext_k,
      surface_faces_up=False,
  )


def _solve_wall_sensor_temp_k(
    vf_vec: np.ndarray,
    radiosity_w_m2: np.ndarray,
    eps_wall: float,
    u_wall: float,
    t_amb_k: float,
    t_ext_k: float,
) -> float:
  """Solves a vertical bare-AAC corner wall point sensor energy balance."""
  return _solve_point_sensor_balance_k(
      vf_vec=vf_vec,
      radiosity_w_m2=radiosity_w_m2,
      eps_surface=eps_wall,
      u_backing=u_wall,
      t_amb_k=t_amb_k,
      t_ext_k=t_ext_k,
      h_conv_fn=lambda dt: max(
          constants.H_CONV_MIN_W_M2K,
          constants.H_CONV_VERT_SENSOR_COEFF * (abs(dt) ** 0.25),
      ),
  )


def compute_safety_metrics(
    mesh: mesh_lib.SurfaceMesh,
    vf_sys: vf_lib.ViewFactorSystem,
    solution: solver_lib.SolverSolution,
    params: params_lib.SimParams,
) -> kiln_sim_pb2.SimulationOutput.SafetyMetrics:
  """Computes all engineering safety metrics and pass/fail flags."""
  stypes = mesh.surface_types
  t_front_c = solution.t_front_k - params_lib.KELVIN_OFFSET

  # 1. Gypsum peak temperatures on left and back walls
  gyp_left_mask = stypes == int(mesh_lib.SurfaceType.GYPSUM_LEFT)
  gyp_back_mask = stypes == int(mesh_lib.SurfaceType.GYPSUM_BACK)
  t_gyp_left_max_c = float(np.max(t_front_c[gyp_left_mask]))
  t_gyp_back_max_c = float(np.max(t_front_c[gyp_back_mask]))
  gypsum_safe = (
      (t_gyp_left_max_c < LIMIT_GYPSUM_C)
      and (t_gyp_back_max_c < LIMIT_GYPSUM_C)
  )

  # 2. SSR enclosure internal air and external casing temperatures
  t_ssr_air_c = solution.t_ssr_air_k - params_lib.KELVIN_OFFSET
  t_ssr_case_c = solution.t_ssr_case_k - params_lib.KELVIN_OFFSET
  ssr_safe = (
      (t_ssr_air_c < LIMIT_SSR_AIR_C) and (t_ssr_case_c < LIMIT_SSR_CASE_C)
  )

  # 3 & 4. Subfloor 2D Lateral Conduction & Kiln Stand Leg Fin Bridging
  r_vip = params.t_vip / params.k_vip
  r_screed = params.t_screed / params.k_screed
  r_pad = (
      params.t_floor_pad / params.k_pad if params.t_floor_pad > 0.0 else 0.0
  )
  r_sub_total = r_pad + r_screed + r_vip
  u_pad_subfloor = 1.0 / r_sub_total
  u_vinyl_subfloor = 1.0 / (r_screed + r_vip)

  if vf_sys.floor_grid_vf is not None and vf_sys.floor_grid_x_m is not None:
    t_surf_2d_k, t_vip_2d_k, in_pad_2d, under_ref_2d = _solve_floor_2d_field(
        mesh=mesh, vf_sys=vf_sys, solution=solution, params=params
    )
    xs = vf_sys.floor_grid_x_m
    ys = vf_sys.floor_grid_y_m
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    r_grid = np.hypot(xx - params.x_kiln_center, yy - params.y_kiln_center)

    r_core = constants.FLOOR_UNDER_CORE_FRACTION * min(
        params.r_kiln, params.r_reflector
    )
    under_mask = (
        (in_pad_2d & (r_grid <= r_core))
        if params.r_reflector > 0.0 and np.any(in_pad_2d & (r_grid <= r_core))
        else (
            (under_ref_2d & in_pad_2d)
            if params.r_reflector > 0.0 and np.any(under_ref_2d & in_pad_2d)
            else (
                in_pad_2d
                & (
                    r_grid
                    <= max(0.5 * params.r_kiln, constants.FLOOR_2D_CELL_SIZE_M)
                )
            )
        )
    )
    if not np.any(under_mask):
      under_mask = np.zeros_like(in_pad_2d)
      under_mask.flat[int(np.argmin(r_grid))] = True

    t_pad_center_k = float(np.max(t_surf_2d_k[under_mask]))
    t_floor_pad_under_reflector_c = t_pad_center_k - params_lib.KELVIN_OFFSET
    t_vip_under_reflector_c = (
        float(np.max(t_vip_2d_k[under_mask])) - params_lib.KELVIN_OFFSET
    )

    pad_out_mask = (
        (in_pad_2d & (~under_ref_2d))
        if params.r_reflector > 0.0 and np.any(in_pad_2d & (~under_ref_2d))
        else (in_pad_2d & (~under_mask))
    )
    if not np.any(pad_out_mask):
      pad_out_mask = in_pad_2d
    t_floor_pad_edge_c = (
        float(np.max(t_surf_2d_k[pad_out_mask])) - params_lib.KELVIN_OFFSET
    )

    vip_out_mask = (
        (~under_ref_2d)
        if params.r_reflector > 0.0 and np.any(~under_ref_2d)
        else (~under_mask)
    )
    if not np.any(vip_out_mask):
      vip_out_mask = np.ones_like(under_mask, dtype=bool)
    t_vip_outside_reflector_c = (
        float(np.max(t_vip_2d_k[vip_out_mask])) - params_lib.KELVIN_OFFSET
    )

    vinyl_mask = ~in_pad_2d
    if np.any(vinyl_mask):
      t_vinyl_edge_c = (
          float(np.max(t_surf_2d_k[vinyl_mask])) - params_lib.KELVIN_OFFSET
      )
    else:
      t_vinyl_x_k = _solve_floor_sensor_temp_k(
          vf_vec=vf_sys.sensor_vf["vinyl_edge_x"],
          radiosity_w_m2=solution.radiosity_w_m2,
          eps_floor=VINYL_FLOOR_EMISSIVITY,
          u_subfloor=u_vinyl_subfloor,
          t_amb_k=solution.room.t_room_lower_k,
          t_ext_k=params.t_ext_k,
      )
      t_vinyl_y_k = _solve_floor_sensor_temp_k(
          vf_vec=vf_sys.sensor_vf["vinyl_edge_y"],
          radiosity_w_m2=solution.radiosity_w_m2,
          eps_floor=VINYL_FLOOR_EMISSIVITY,
          u_subfloor=u_vinyl_subfloor,
          t_amb_k=solution.room.t_room_lower_k,
          t_ext_k=params.t_ext_k,
      )
      t_vinyl_edge_c = max(t_vinyl_x_k, t_vinyl_y_k) - params_lib.KELVIN_OFFSET
  else:
    ref_mask = stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
    if params.r_reflector <= 0.0:
      t_pad_center_k = _solve_floor_sensor_temp_k(
          vf_vec=vf_sys.sensor_vf["pad_center"],
          radiosity_w_m2=solution.radiosity_w_m2,
          eps_floor=FLOOR_PAD_EMISSIVITY,
          u_subfloor=u_pad_subfloor,
          t_amb_k=solution.room.t_room_lower_k,
          t_ext_k=params.t_ext_k,
      )
    else:
      _, _, _, t_pad_center_k = solver_lib.solve_under_reflector_column(
          t_ref_front_k=float(solution.t_front_k[ref_mask][0]),
          t_room_lower_k=solution.room.t_room_lower_k,
          params=params,
      )

    t_floor_pad_under_reflector_c = t_pad_center_k - params_lib.KELVIN_OFFSET
    t_vip_under_ref_k = params.t_ext_k + (
        (t_pad_center_k - params.t_ext_k) * (r_vip / r_sub_total)
    )
    t_vip_under_reflector_c = t_vip_under_ref_k - params_lib.KELVIN_OFFSET

    pad_sensor_names = [
        name
        for name in vf_sys.sensor_vf
        if name in ("pad_edge_corner", "pad_edge_room")
        or name.startswith("pad_outside_")
    ]
    pad_outside_temps_k = [
        _solve_floor_sensor_temp_k(
            vf_vec=vf_sys.sensor_vf[name],
            radiosity_w_m2=solution.radiosity_w_m2,
            eps_floor=FLOOR_PAD_EMISSIVITY,
            u_subfloor=u_pad_subfloor,
            t_amb_k=solution.room.t_room_lower_k,
            t_ext_k=params.t_ext_k,
        )
        for name in pad_sensor_names
    ]
    t_pad_edge_max_k = max(pad_outside_temps_k)
    t_floor_pad_edge_c = t_pad_edge_max_k - params_lib.KELVIN_OFFSET

    t_vip_outside_ref_k = params.t_ext_k + (
        (t_pad_edge_max_k - params.t_ext_k) * (r_vip / r_sub_total)
    )
    t_vip_outside_reflector_c = t_vip_outside_ref_k - params_lib.KELVIN_OFFSET

    t_vinyl_x_k = _solve_floor_sensor_temp_k(
        vf_vec=vf_sys.sensor_vf["vinyl_edge_x"],
        radiosity_w_m2=solution.radiosity_w_m2,
        eps_floor=VINYL_FLOOR_EMISSIVITY,
        u_subfloor=u_vinyl_subfloor,
        t_amb_k=solution.room.t_room_lower_k,
        t_ext_k=params.t_ext_k,
    )
    t_vinyl_y_k = _solve_floor_sensor_temp_k(
        vf_vec=vf_sys.sensor_vf["vinyl_edge_y"],
        radiosity_w_m2=solution.radiosity_w_m2,
        eps_floor=VINYL_FLOOR_EMISSIVITY,
        u_subfloor=u_vinyl_subfloor,
        t_amb_k=solution.room.t_room_lower_k,
        t_ext_k=params.t_ext_k,
    )
    t_vinyl_edge_c = max(t_vinyl_x_k, t_vinyl_y_k) - params_lib.KELVIN_OFFSET

  floor_under_reflector_safe = t_vip_under_reflector_c < LIMIT_VIP_C
  floor_outside_reflector_safe = (
      (t_vip_outside_reflector_c < LIMIT_VIP_C)
      and (t_vinyl_edge_c < LIMIT_VINYL_C)
  )
  floor_safe = floor_under_reflector_safe and floor_outside_reflector_safe

  # 5. Shield, AAC wall, and behind-shield chimney peak metrics
  shield_mask = (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT)) | (
      stypes == int(mesh_lib.SurfaceType.SHIELD_BACK)
  )
  t_shield_max_c = float(np.max(t_front_c[shield_mask]))
  aac_candidates_k = [
      float(np.max(solution.chimney.t_aac_front_patch_k[shield_mask])),
      float(np.max(solution.t_back_k[gyp_left_mask | gyp_back_mask])),
  ]
  u_aac_bare = 1.0 / ((params.t_aac / params.k_aac) + params_lib.R_SE_WALL)
  for aac_name, (aac_coord, _) in vf_lib.bare_aac_sensor_points(params).items():
    if aac_name in vf_sys.sensor_vf:
      z_pt = aac_coord[2]
      t_amb_aac_k = (
          solution.room.t_room_upper_k
          if z_pt >= solution.room.z_stratification_layer_m
          else solution.room.t_room_lower_k
      )
      aac_candidates_k.append(
          _solve_wall_sensor_temp_k(
              vf_vec=vf_sys.sensor_vf[aac_name],
              radiosity_w_m2=solution.radiosity_w_m2,
              eps_wall=params.eps_aac,
              u_wall=u_aac_bare,
              t_amb_k=t_amb_aac_k,
              t_ext_k=params.t_ext_k,
          )
      )
  t_aac_max_c = max(aac_candidates_k) - params_lib.KELVIN_OFFSET
  v_chimney_max = max(
      float(np.max(solution.chimney.v_gap_left)),
      float(np.max(solution.chimney.v_gap_back)),
  )
  t_chimney_exit_max_k = max(
      float(np.max(solution.chimney.t_exit_left_k)),
      float(np.max(solution.chimney.t_exit_back_k)),
  )
  t_chimney_exit_max_c = t_chimney_exit_max_k - params_lib.KELVIN_OFFSET

  # 6. Ceiling peak surface temperature across z = h_wall
  uncaptured_chimney_frac = 1.0 - solution.room.chimney_capture_fraction
  dt_chim_excess = max(0.0, t_chimney_exit_max_k - solution.room.t_room_upper_k)
  uncaptured_plume_frac = 1.0 - solution.room.plume_capture_fraction
  dt_plume_excess = max(
      0.0, solution.room.t_plume_at_hood_k - solution.room.t_room_upper_k
  )
  r_0_plume = max(
      0.15, params.r_hood_inlet + constants.CEILING_SENSOR_RIM_OFFSET_M
  )

  ceil_vfs: list[np.ndarray] = []
  ceil_ambs_k: list[float] = []
  for name, (coord, _) in vf_lib.ceiling_sensor_points(params).items():
    if name not in vf_sys.sensor_vf:
      continue
    xc, yc, _ = coord
    r_h = math.hypot(xc - params.x_hood, yc - params.y_hood)
    if r_h < params.r_hood_inlet:
      f_plume_r = math.exp(
          -(params.r_hood_inlet - r_h) / constants.CEILING_PLUME_INNER_DECAY_M
      )
    else:
      f_plume_r = (max(r_h, r_0_plume) / r_0_plume) ** (-2.0 / 3.0)
    dt_plume_jet = (
        constants.PLUME_CEILING_MIXING_FRACTION
        * uncaptured_plume_frac
        * dt_plume_excess
        * f_plume_r
    )

    d_left = math.hypot(
        max(0.0, xc - params.x_shield_left_end),
        max(0.0, yc - params.y_shield_front_left_wall),
    )
    d_back = math.hypot(
        max(0.0, xc - params.x_shield_front_back_wall),
        max(0.0, yc - params.y_shield_back_end),
    )
    d_chim = max(
        0.0, min(d_left, d_back) - constants.CEILING_SENSOR_WALL_OFFSET_M
    )
    dt_chim_jet = (
        CHIMNEY_CEILING_MIXING_FRACTION
        * uncaptured_chimney_frac
        * dt_chim_excess
        * math.exp(-d_chim / constants.CEILING_CHIMNEY_JET_DECAY_M)
    )
    ceil_vfs.append(vf_sys.sensor_vf[name])
    ceil_ambs_k.append(
        solution.room.t_room_upper_k + max(dt_plume_jet, dt_chim_jet)
    )

  if ceil_vfs:
    ceil_temps_k = _solve_ceiling_sensors_vec_k(
        vf_mat=np.vstack(ceil_vfs),
        radiosity_w_m2=solution.radiosity_w_m2,
        eps_ceil=params.eps_gypsum,
        u_roof=params.u_roof,
        t_amb_vec_k=np.array(ceil_ambs_k, dtype=np.float64),
        t_ext_k=params.t_ext_k,
    )
    t_ceiling_max_c = float(np.max(ceil_temps_k)) - params_lib.KELVIN_OFFSET
  else:
    t_ceiling_max_c = solution.room.t_room_upper_k - params_lib.KELVIN_OFFSET
  ceiling_safe = t_ceiling_max_c < LIMIT_CEILING_C

  # 6b. Exhaust duct mixed air temperature (including teed bung exhaust)
  q_bung_duct_w = (
      max(0.0, params.q_bung_open_w) if params.bung_captured else 0.0
  )
  m_dot_hood = constants.AIR_RHO_REF * solution.room.v_dot_hood_eff_m3_s
  if m_dot_hood > constants.MIN_DUCT_MASS_FLOW_KG_S:
    t_exhaust_duct_k = params.t_ext_k + (
        solution.room.q_hood_exhaust_w + q_bung_duct_w
    ) / (m_dot_hood * constants.AIR_CP)
  else:
    t_exhaust_duct_k = solution.room.t_room_upper_k
  t_exhaust_duct_c = t_exhaust_duct_k - params_lib.KELVIN_OFFSET

  # 7. Kiln azimuthal skin temperature spread
  kiln_side_mask = stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
  t_kiln_side_c = t_front_c[kiln_side_mask]
  delta_t_kiln_skin_c = float(np.max(t_kiln_side_c) - np.min(t_kiln_side_c))

  return kiln_sim_pb2.SimulationOutput.SafetyMetrics(
      t_gypsum_max_left_c=t_gyp_left_max_c,
      t_gypsum_max_back_c=t_gyp_back_max_c,
      gypsum_safe=gypsum_safe,
      t_ssr_air_c=t_ssr_air_c,
      t_ssr_case_c=t_ssr_case_c,
      ssr_safe=ssr_safe,
      t_vip_under_reflector_c=t_vip_under_reflector_c,
      t_floor_pad_under_reflector_c=t_floor_pad_under_reflector_c,
      floor_under_reflector_safe=floor_under_reflector_safe,
      t_vip_outside_reflector_c=t_vip_outside_reflector_c,
      t_floor_pad_edge_c=t_floor_pad_edge_c,
      t_vinyl_edge_c=t_vinyl_edge_c,
      floor_outside_reflector_safe=floor_outside_reflector_safe,
      floor_safe=floor_safe,
      t_shield_max_c=t_shield_max_c,
      t_aac_max_c=t_aac_max_c,
      v_chimney_max_m_s=v_chimney_max,
      t_chimney_exit_max_c=t_chimney_exit_max_c,
      plume_fully_captured=solution.room.plume_fully_captured,
      plume_capture_fraction=solution.room.plume_capture_fraction,
      chimney_capture_fraction=solution.room.chimney_capture_fraction,
      t_room_upper_c=solution.room.t_room_upper_k - params_lib.KELVIN_OFFSET,
      t_room_lower_c=solution.room.t_room_lower_k - params_lib.KELVIN_OFFSET,
      z_stratification_layer_mm=(
          solution.room.z_stratification_layer_m * params_lib.M_TO_MM
      ),
      delta_t_kiln_skin_c=delta_t_kiln_skin_c,
      t_ceiling_max_c=t_ceiling_max_c,
      ceiling_safe=ceiling_safe,
      t_exhaust_duct_c=t_exhaust_duct_c,
  )


def build_simulation_output(
    config_pb: kiln_sim_pb2.SimulationConfig,
    input_config_path: str,
    input_config_md5: str,
    timestamp_utc: str,
    params: params_lib.SimParams,
    mesh: mesh_lib.SurfaceMesh,
    vf_sys: vf_lib.ViewFactorSystem,
    solution: solver_lib.SolverSolution,
    include_patches: bool = True,
) -> kiln_sim_pb2.SimulationOutput:
  """Assembles the complete SimulationOutput protobuf record."""
  provenance = kiln_sim_pb2.SimulationOutput.Provenance(
      input_config_path=input_config_path,
      input_config_md5=input_config_md5,
      timestamp_utc=timestamp_utc,
      solver_wall_time_s=solution.solver_wall_time_s,
      converged=solution.converged,
      max_energy_residual_w_m2=solution.max_energy_residual_w_m2,
  )
  derived_pb = params.to_derived_proto()
  metrics_pb = compute_safety_metrics(
      mesh=mesh, vf_sys=vf_sys, solution=solution, params=params
  )
  patches_pb = (
      mesh.to_patch_protos(solution=solution) if include_patches else []
  )

  output_pb = kiln_sim_pb2.SimulationOutput(
      provenance=provenance,
      derived=derived_pb,
      metrics=metrics_pb,
      patches=patches_pb,
  )
  output_pb.input_config_snapshot.CopyFrom(config_pb)
  return output_pb