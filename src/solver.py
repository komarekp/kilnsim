"""Coupled radiative, convective, conductive, and room-zone thermal solver."""

import dataclasses
import math
import time
from typing import Any
from absl import logging
import numpy as np
from scipy import linalg
from src import chimney as chimney_lib
from src import constants
from src import mesh as mesh_lib
from src import params as params_lib
from src import plume_room as plume_room_lib
from src import view_factors as vf_lib


@dataclasses.dataclass(frozen=True)
class SolverSolution:
  """Converged steady-state thermal and airflow solution across all domains.

  Attributes:
    t_front_k: (N_patches,) converged room-facing surface temperatures (K).
    t_back_k: (N_patches,) converged back-face / wall-interface temperatures (K).
    radiosity_w_m2: (N_patches,) converged surface radiosity J_m (W/m^2).
    q_rad_net_w_m2: (N_patches,) net radiative heat flux leaving front face (W/m^2).
    q_conv_net_w_m2: (N_patches,) net convective heat flux leaving front face (W/m^2).
    q_cond_net_w_m2: (N_patches,) conductive heat flux entering front face from
      interior or leaving through wall (W/m^2).
    chimney: Converged ChimneySolution instance.
    room: Converged RoomSolution instance.
    t_ssr_air_k: Converged internal air temperature inside the SSR housing (K).
    t_ssr_case_k: Area-averaged SSR sheet-metal enclosure temperature (K).
    converged: True if both inner surface solver and outer room loop converged.
    max_energy_residual_w_m2: Peak absolute patch energy balance residual (W/m^2).
    solver_wall_time_s: Total wall-clock solve duration in seconds.
  """

  t_front_k: np.ndarray
  t_back_k: np.ndarray
  radiosity_w_m2: np.ndarray
  q_rad_net_w_m2: np.ndarray
  q_conv_net_w_m2: np.ndarray
  q_cond_net_w_m2: np.ndarray
  chimney: chimney_lib.ChimneySolution
  room: plume_room_lib.RoomSolution
  t_ssr_air_k: float
  t_ssr_case_k: float
  converged: bool
  max_energy_residual_w_m2: float
  solver_wall_time_s: float
  dt_room_change_k: float = 0.0


def compute_natural_convection_h(
    t_surf_k: np.ndarray,
    t_amb_k: np.ndarray,
    char_lengths_m: np.ndarray,
    surface_types: np.ndarray,
) -> np.ndarray:
  """Computes Churchill-Chu and horizontal-plate natural convection coefficients.

  Args:
    t_surf_k: (N,) surface temperatures in Kelvin.
    t_amb_k: (N,) local ambient air temperatures in Kelvin.
    char_lengths_m: (N,) characteristic buoyancy lengths L in metres.
    surface_types: (N,) integer SurfaceType enum values.

  Returns:
    (N,) convective heat transfer coefficients h_c in W/(m^2*K).
  """
  t_film = np.maximum(constants.SOLVER_MIN_TEMP_K, 0.5 * (t_surf_k + t_amb_k))
  beta = 1.0 / t_film
  delta_t = np.sqrt((t_surf_k - t_amb_k) ** 2 + 1e-6)

  nu = constants.AIR_NU_REF_300K * (
      (t_film / constants.AIR_T_REF_K) ** 1.7
  )
  alpha = nu / constants.AIR_PR
  k_air = constants.AIR_K_REF_300K * (
      (t_film / constants.AIR_T_REF_K) ** 0.8
  )

  ra_l = (
      constants.GRAVITY
      * beta
      * delta_t
      * (char_lengths_m ** 3)
      / np.maximum(nu * alpha, 1e-14)
  )

  pr_term = (1.0 + (0.492 / constants.AIR_PR) ** (9.0 / 16.0)) ** (
      8.0 / 27.0
  )
  nu_vert = (0.825 + (0.387 * (ra_l ** (1.0 / 6.0))) / pr_term) ** 2
  h_c = (nu_vert * k_air) / np.maximum(char_lengths_m, 0.05)

  up_mask = (surface_types == int(mesh_lib.SurfaceType.KILN_LID)) | (
      surface_types == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  )
  if np.any(up_mask):
    nu_up = 0.15 * np.maximum(ra_l[up_mask], 1.0e4) ** (1.0 / 3.0)
    h_c[up_mask] = (nu_up * k_air[up_mask]) / np.maximum(
        char_lengths_m[up_mask], 0.05
    )

  bot_mask = surface_types == int(mesh_lib.SurfaceType.KILN_BOTTOM)
  if np.any(bot_mask):
    nu_bot = 0.27 * np.maximum(ra_l[bot_mask], 1.0e4) ** 0.25
    h_c[bot_mask] = (nu_bot * k_air[bot_mask]) / np.maximum(
        char_lengths_m[bot_mask], 0.05
    )

  return np.maximum(constants.H_CONV_MIN_W_M2K, h_c)


def _diffuse_reflectances(mesh: mesh_lib.SurfaceMesh) -> np.ndarray:
  """Returns per-patch diffuse reflectances (zero on specular REFLECTOR_TOP)."""
  is_ref = mesh.surface_types == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  return np.where(is_ref, 0.0, 1.0 - mesh.emissivities)


def solve_radiosity_system(
    t_front_k: np.ndarray,
    t_amb_rad_k: np.ndarray,
    lu_piv: tuple[np.ndarray, np.ndarray],
    mesh: mesh_lib.SurfaceMesh,
    vf_sys: vf_lib.ViewFactorSystem,
) -> tuple[np.ndarray, np.ndarray]:
  """Solves the linear radiosity matrix equation for net outgoing radiative flux.

  Args:
    t_front_k: (N,) surface front temperatures in Kelvin.
    t_amb_rad_k: (N,) effective ambient background radiation temperatures (K).
    lu_piv: Pre-factored LU decomposition of `I - rho_d * F_eff`.
    mesh: Discretized SurfaceMesh instance.
    vf_sys: Precomputed ViewFactorSystem instance.

  Returns:
    Tuple of (radiosity J in W/m^2, net outgoing radiative flux q_rad_out in
    W/m^2 where positive means net radiation leaving the surface).
  """
  eps = mesh.emissivities
  rho_d = _diffuse_reflectances(mesh)
  is_ref = mesh.surface_types == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  eb_surf = params_lib.STEFAN_BOLTZMANN * (t_front_k ** 4)
  eb_room = params_lib.STEFAN_BOLTZMANN * (t_amb_rad_k ** 4)

  eb_term = np.where(is_ref, eb_surf, eps * eb_surf)
  rhs = eb_term + rho_d * vf_sys.f_room * eb_room
  radiosity = linalg.lu_solve(lu_piv, rhs)
  irr = vf_sys.f_eff @ radiosity + vf_sys.f_room * eb_room
  q_rad_out = eps * (eb_surf - irr)
  return radiosity, q_rad_out


def _solve_ssr_internal_air(
    t_ssr_faces_k: np.ndarray,
    ssr_areas: np.ndarray,
    t_ref_k: float,
    t_lower_k: float,
    params: params_lib.SimParams,
    t_kiln_facing_k: float | None = None,
    q_bot_conv_w: float = 0.0,
) -> tuple[float, float]:
  """Computes the steady-state SSR internal air and 4-wall casing temperatures."""
  a_outer_3 = float(np.sum(ssr_areas))
  sum_t_a_3 = float(np.sum(t_ssr_faces_k * ssr_areas))
  t_ssr_case_3_k = sum_t_a_3 / a_outer_3

  a_ref = math.pi * (params.r_reflector ** 2)
  if params.r_reflector > 0.0:
    h_ref = float(
        compute_natural_convection_h(
            t_surf_k=np.array([t_ref_k], dtype=np.float64),
            t_amb_k=np.array([t_lower_k], dtype=np.float64),
            char_lengths_m=np.array([params.d_reflector], dtype=np.float64),
            surface_types=np.array(
                [int(mesh_lib.SurfaceType.REFLECTOR_TOP)], dtype=np.int32
            ),
        )[0]
    )
  else:
    h_ref = 0.0
  q_ref_conv = max(0.0, h_ref * a_ref * (t_ref_k - t_lower_k))
  q_subkiln_conv = (
      q_ref_conv + constants.SSR_SUBKILN_CONV_WEIGHT * max(0.0, q_bot_conv_w)
  )
  m_dot_local_inflow = constants.SSR_LOCAL_INFLOW_KG_S
  dt_floor_conv = q_subkiln_conv / (m_dot_local_inflow * constants.AIR_CP)
  t_ssr_inlet_k = t_lower_k + dt_floor_conv

  a_louver_eff = constants.SSR_LOUVER_EFF_AREA_M2
  h_int_box = constants.H_CONV_SSR_INTERNAL_W_M2K
  a_ssr_4 = params.w_ssr * params.h_ssr
  a_box_total = a_outer_3 + a_ssr_4
  ua_box = h_int_box * a_box_total

  eps_k = max(0.05, params.eps_kiln_side)
  eps_s = max(0.05, params.eps_ssr)
  sigma = constants.STEFAN_BOLTZMANN
  c_rad_kb = sigma / ((1.0 / eps_k) + (1.0 / eps_s) - 1.0)
  c_rad_b4 = sigma / ((2.0 / eps_s) - 1.0)
  h_baf = constants.SSR_BAFFLE_CONV_W_M2K
  t_k_ssr = t_lower_k if t_kiln_facing_k is None else float(t_kiln_facing_k)

  t_baf_k = 0.5 * (t_k_ssr + t_lower_k)
  t_ssr_4_k = 0.5 * (t_ssr_case_3_k + t_baf_k)
  t_air_k = 0.5 * (t_ssr_inlet_k + t_ssr_case_3_k) + (
      params.q_ssr_internal_w / max(ua_box, 1.0)
  )
  t_ssr_case_k = t_ssr_case_3_k
  for _ in range(8):
    for _ in range(3):
      t_k4 = t_k_ssr ** 4
      t_b3 = t_baf_k ** 3
      t_b4 = t_baf_k * t_b3
      t_43 = t_ssr_4_k ** 3
      t_44 = t_ssr_4_k * t_43
      f1 = (
          c_rad_kb * (t_k4 - t_b4)
          - c_rad_b4 * (t_b4 - t_44)
          - 2.0 * h_baf * (t_baf_k - t_lower_k)
      )
      f2 = (
          c_rad_b4 * (t_b4 - t_44)
          + h_baf * (t_lower_k - t_ssr_4_k)
          - h_int_box * (t_ssr_4_k - t_air_k)
      )
      a11 = 4.0 * (c_rad_kb + c_rad_b4) * t_b3 + 2.0 * h_baf
      a12 = 4.0 * c_rad_b4 * t_43
      a21 = 4.0 * c_rad_b4 * t_b3
      a22 = 4.0 * c_rad_b4 * t_43 + h_baf + h_int_box
      det = max(a11 * a22 - a12 * a21, 1e-12)
      dt_b = max(-50.0, min(50.0, (a22 * f1 + a12 * f2) / det))
      dt_4 = max(-50.0, min(50.0, (a21 * f1 + a11 * f2) / det))
      t_baf_k = max(200.0, t_baf_k + dt_b)
      t_ssr_4_k = max(200.0, t_ssr_4_k + dt_4)

    t_ssr_case_k = (sum_t_a_3 + t_ssr_4_k * a_ssr_4) / a_box_total
    dt_buoy = abs(t_air_k - t_ssr_inlet_k)
    v_louver = constants.SSR_LOUVER_DISCHARGE_COEFF * math.sqrt(
        2.0 * constants.GRAVITY * params.h_ssr * (dt_buoy / t_ssr_inlet_k)
    )
    m_dot_louver = max(
        constants.SSR_LOUVER_MIN_FLOW_KG_S,
        constants.AIR_RHO_REF * a_louver_eff * v_louver,
    )
    mcp = m_dot_louver * constants.AIR_CP
    t_air_new = (
        params.q_ssr_internal_w + mcp * t_ssr_inlet_k + ua_box * t_ssr_case_k
    ) / (mcp + ua_box)
    t_air_k = 0.5 * t_air_k + 0.5 * t_air_new

  return t_air_k, t_ssr_case_k


def solve_under_reflector_column(
    t_ref_front_k: float,
    t_room_lower_k: float,
    params: params_lib.SimParams,
) -> tuple[float, float, float, float]:
  """Solves 1D steady heat transfer from reflector front face through subfloor.

  Accounts for optional CalSil backing conduction, bare stainless vs. CalSil
  underside emissivity across the elevated gap, under-reflector convection to
  `t_room_lower_k`, and subfloor conduction (pad + screed + VIP) to `t_ext_k`.

  Args:
    t_ref_front_k: Reflector top surface temperature (K).
    t_room_lower_k: Lower-zone room air temperature (K).
    params: Validated SimParams instance.

  Returns:
    Tuple `(q_down_w_m2, du_down_w_m2k, t_ref_back_k, t_pad_center_k)` where
    `q_down_w_m2` is net heat flux leaving the reflector downward (W/m^2),
    `du_down_w_m2k` is its local linearization `dq_down / dT_ref_front`,
    `t_ref_back_k` is the reflector underside temperature (K), and
    `t_pad_center_k` is the floor pad / screed top temperature directly beneath
    the reflector center (K).
  """
  r_pad = (
      params.t_floor_pad / max(params.k_pad, 0.1)
      if params.t_floor_pad > 0.0
      else 0.0
  )
  r_sub = (
      r_pad
      + (params.t_screed / params.k_screed)
      + (params.t_vip / params.k_vip)
  )
  u_sub = 1.0 / r_sub
  has_calsil = params.t_reflector_calsil > constants.GEOM_TOL_M
  r_calsil = (params.t_reflector_calsil / params.k_shield) if has_calsil else 0.0

  if params.g_floor_reflector <= constants.GEOM_TOL_M:
    u_total = 1.0 / (r_calsil + r_sub)
    q_down = u_total * (t_ref_front_k - params.t_ext_k)
    t_ref_back_k = t_ref_front_k - q_down * r_calsil
    return q_down, u_total, t_ref_back_k, t_ref_back_k

  eps_underside = (
      params.eps_shield if has_calsil else params.eps_reflector_top
  )
  eps_gap = 1.0 / (
      (1.0 / eps_underside) + (1.0 / constants.FLOOR_PAD_EMISSIVITY) - 1.0
  )
  h_under = constants.H_CONV_UNDER_REFLECTOR_W_M2K
  sigma = constants.STEFAN_BOLTZMANN

  t_rb = float(t_ref_front_k)
  t_p = 0.5 * (t_rb + t_room_lower_k)
  g_calsil = (1.0 / r_calsil) if has_calsil else 0.0

  for _ in range(10):
    f_pad = (
        eps_gap * sigma * ((t_rb ** 4) - (t_p ** 4))
        - h_under * (t_p - t_room_lower_k)
        - u_sub * (t_p - params.t_ext_k)
    )
    h_rad_b = 4.0 * eps_gap * sigma * (t_rb ** 3)
    h_rad_p = 4.0 * eps_gap * sigma * (t_p ** 3)
    df_pad_dtp = -(h_rad_p + h_under + u_sub)

    if has_calsil:
      f_rb = (
          g_calsil * (t_ref_front_k - t_rb)
          - eps_gap * sigma * ((t_rb ** 4) - (t_p ** 4))
          - h_under * (t_rb - t_room_lower_k)
      )
      df_rb_dtrb = -(g_calsil + h_rad_b + h_under)
      df_rb_dtp = h_rad_p
      df_pad_dtrb = h_rad_b
      det = df_rb_dtrb * df_pad_dtp - df_rb_dtp * df_pad_dtrb
      dt_rb = (-f_rb * df_pad_dtp + f_pad * df_rb_dtp) / det
      dt_p = (-df_rb_dtrb * f_pad + df_pad_dtrb * f_rb) / det
      t_rb = max(constants.SOLVER_MIN_TEMP_K, t_rb + dt_rb)
      t_p = max(constants.SOLVER_MIN_TEMP_K, t_p + dt_p)
      if max(abs(dt_rb), abs(dt_p)) < 1e-8:
        break
    else:
      dt_p = -f_pad / df_pad_dtp
      t_p = max(constants.SOLVER_MIN_TEMP_K, t_p + dt_p)
      if abs(dt_p) < 1e-8:
        break

  q_rad_gap = eps_gap * sigma * ((t_rb ** 4) - (t_p ** 4))
  q_down = q_rad_gap + h_under * (t_rb - t_room_lower_k)

  h_rad_b = 4.0 * eps_gap * sigma * (t_rb ** 3)
  h_rad_p = 4.0 * eps_gap * sigma * (t_p ** 3)
  g_gap_eff = h_under + h_rad_b * (h_under + u_sub) / (
      h_rad_p + h_under + u_sub
  )
  du_down = (
      (g_calsil * g_gap_eff) / (g_calsil + g_gap_eff)
      if has_calsil
      else g_gap_eff
  )
  return float(q_down), float(du_down), float(t_rb), float(t_p)


def factor_radiosity_matrix(
    mesh: mesh_lib.SurfaceMesh,
    vf_sys: vf_lib.ViewFactorSystem,
) -> tuple[np.ndarray, np.ndarray]:
  """Precomputes the LU factorization of `I - rho_d * F_eff`."""
  n_patches = mesh.num_patches
  rho_d = _diffuse_reflectances(mesh)
  rad_matrix = np.eye(n_patches, dtype=np.float64) - (
      rho_d[:, None] * vf_sys.f_eff
  )
  return linalg.lu_factor(rad_matrix)


def solve_thermal_system(
    mesh: mesh_lib.SurfaceMesh,
    vf_sys: vf_lib.ViewFactorSystem,
    params: params_lib.SimParams,
    lu_piv: tuple[np.ndarray, np.ndarray] | None = None,
    initial_solution: SolverSolution | None = None,
) -> SolverSolution:
  """Executes the coupled inner (surface) and middle (room) solver loops.

  Uses a vectorized diagonal-Newton solver with exact analytical local
  derivatives `dR_m/dT_m` and full radiosity/chimney updates at each step,
  avoiding O(N^2) finite-difference Jacobian evaluations.

  Args:
    mesh: Discretized SurfaceMesh instance.
    vf_sys: Precomputed ViewFactorSystem instance.
    params: Validated SimParams instance.
    lu_piv: Optional precomputed LU factorization from `factor_radiosity_matrix`.
    initial_solution: Optional converged SolverSolution from a similar run to
      warm-start surface and room zone temperatures.

  Returns:
    Converged SolverSolution instance.
  """
  t_start = time.monotonic()
  n_patches = mesh.num_patches
  stypes = mesh.surface_types
  z_coords = mesh.centroids[:, 2]
  eps = mesh.emissivities

  if lu_piv is None:
    lu_piv = factor_radiosity_matrix(mesh, vf_sys)

  char_lengths = np.full(n_patches, params.h_wall, dtype=np.float64)
  char_lengths[stypes == int(mesh_lib.SurfaceType.KILN_SIDE)] = (
      params.h_kiln_total
  )
  char_lengths[
      (stypes == int(mesh_lib.SurfaceType.KILN_LID))
      | (stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM))
  ] = (0.5 * params.r_kiln)
  char_lengths[stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)] = max(
      0.05, 0.5 * params.r_reflector
  )
  char_lengths[
      (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT))
      | (stypes == int(mesh_lib.SurfaceType.SHIELD_BACK))
  ] = params.h_shield
  char_lengths[stypes == int(mesh_lib.SurfaceType.SSR_BOX)] = params.h_ssr

  kiln_side_mask = stypes == int(mesh_lib.SurfaceType.KILN_SIDE)
  kiln_lid_mask = stypes == int(mesh_lib.SurfaceType.KILN_LID)
  kiln_bot_mask = stypes == int(mesh_lib.SurfaceType.KILN_BOTTOM)
  shield_mask = (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT)) | (
      stypes == int(mesh_lib.SurfaceType.SHIELD_BACK)
  )
  gypsum_mask = (stypes == int(mesh_lib.SurfaceType.GYPSUM_LEFT)) | (
      stypes == int(mesh_lib.SurfaceType.GYPSUM_BACK)
  )
  ref_mask = stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  ssr_mask = stypes == int(mesh_lib.SurfaceType.SSR_BOX)

  u_gypsum_wall = 1.0 / (
      (params.t_gypsum / params.k_gypsum)
      + (params.t_aac / params.k_aac)
      + params_lib.R_SE_WALL
  )
  g_shield = params.k_shield / params.t_shield

  u_cond_diag = np.zeros(n_patches, dtype=np.float64)
  u_cond_diag[kiln_side_mask] = params.u_kiln_side
  u_cond_diag[kiln_lid_mask] = params.u_kiln_top
  u_cond_diag[kiln_bot_mask] = params.u_kiln_bot
  u_cond_diag[gypsum_mask] = u_gypsum_wall
  u_cond_diag[ssr_mask] = constants.H_CONV_SSR_INTERNAL_W_M2K

  dt_firing = max(0.0, params.t_internal_k - params.t_ext_k)
  if (
      initial_solution is not None
      and initial_solution.converged
      and initial_solution.t_front_k.shape == (n_patches,)
  ):
    t_room_lower_k = initial_solution.room.t_room_lower_k
    t_room_upper_k = initial_solution.room.t_room_upper_k
    z_layer_m = initial_solution.room.z_stratification_layer_m
    t_front_k = initial_solution.t_front_k.copy()
    t_back_k = initial_solution.t_back_k.copy()
    t_back_k[kiln_side_mask | kiln_lid_mask | kiln_bot_mask] = (
        params.t_internal_k
    )
  else:
    t_room_lower_k = params.t_ext_k
    t_room_upper_k = params.t_ext_k
    z_layer_m = params.z_hood_inlet

    t_front_k = np.full(
        n_patches, params.t_ext_k + 0.008 * dt_firing, dtype=np.float64
    )
    t_front_k[kiln_side_mask] = params.t_ext_k + 0.095 * dt_firing
    t_front_k[kiln_lid_mask] = params.t_ext_k + 0.130 * dt_firing
    t_front_k[kiln_bot_mask] = params.t_ext_k + 0.120 * dt_firing

    t_back_k = t_front_k.copy()
    t_back_k[kiln_side_mask | kiln_lid_mask | kiln_bot_mask] = (
        params.t_internal_k
    )

  latest_state: dict[str, Any] = {}
  t_max_bound = max(params.t_internal_k + 50.0, params.t_ext_k + 100.0)

  def _evaluate_step(
      t_vec: np.ndarray, update_chimney: bool = True
  ) -> tuple[np.ndarray, np.ndarray]:
    """Computes residual R_m and analytical diagonal Jacobian dR_m/dT_m."""
    t_safe = np.clip(t_vec, 250.0, t_max_bound)
    t_amb_local = np.where(
        z_coords >= z_layer_m, t_room_upper_k, t_room_lower_k
    )

    # 1. Solve linear radiosity system
    radiosity, q_rad_out = solve_radiosity_system(
        t_front_k=t_safe,
        t_amb_rad_k=t_amb_local,
        lu_piv=lu_piv,
        mesh=mesh,
        vf_sys=vf_sys,
    )
    eb_surf = params_lib.STEFAN_BOLTZMANN * (t_safe ** 4)
    eb_room = params_lib.STEFAN_BOLTZMANN * (t_amb_local ** 4)
    q_rad_room_patch_w = np.where(
        ref_mask,
        mesh.areas
        * params.eps_reflector_top
        * vf_sys.f_room
        * (eb_surf - eb_room),
        mesh.areas * vf_sys.f_room * (radiosity - eb_room),
    )

    # 2. Front-face natural convection
    h_front = compute_natural_convection_h(
        t_surf_k=t_safe,
        t_amb_k=t_amb_local,
        char_lengths_m=char_lengths,
        surface_types=stypes,
    )
    q_conv_out = h_front * (t_safe - t_amb_local)

    # 3. Behind-shield 1D chimney solver & Newton update for shield back face
    if update_chimney or "chimney" not in latest_state:
      for _ in range(2):
        chimney_sol = chimney_lib.solve_chimney(
            t_shield_back_k=t_back_k,
            t_inlet_k=t_room_lower_k,
            mesh=mesh,
            params=params,
        )
        q_back = chimney_sol.q_back_from_shield_w_m2[shield_mask]
        h_eff_b = chimney_sol.h_eff_back_patch[shield_mask]
        t_back_k[shield_mask] += (
            g_shield * (t_safe[shield_mask] - t_back_k[shield_mask]) - q_back
        ) / (g_shield + h_eff_b)

      chimney_sol = chimney_lib.solve_chimney(
          t_shield_back_k=t_back_k,
          t_inlet_k=t_room_lower_k,
          mesh=mesh,
          params=params,
      )
    else:
      chimney_sol = latest_state["chimney"]

    h_eff_b = chimney_sol.h_eff_back_patch[shield_mask]
    u_cond_diag[shield_mask] = (g_shield * h_eff_b) / (g_shield + h_eff_b)

    # 4. SSR internal air & casing balance (with 4th kiln-facing wall/baffle
    # and sub-kiln louver intake air preheating)
    t_ref_val = (
        float(t_safe[ref_mask][0]) if np.any(ref_mask) else t_room_lower_k
    )
    t_kiln_facing_k: float | None = None
    if np.any(kiln_side_mask):
      ur_x = math.cos(params.ssr_angle_rad)
      ur_y = math.sin(params.ssr_angle_rad)
      dot_ssr = mesh.normals[:, 0] * ur_x + mesh.normals[:, 1] * ur_y
      facing_mask = (
          kiln_side_mask
          & (dot_ssr >= constants.SSR_FACING_COS_MIN)
          & (mesh.bounds_max[:, 2] > params.z_ssr_bot)
          & (mesh.bounds_min[:, 2] < params.z_ssr_top)
      )
      if not np.any(facing_mask):
        facing_mask = kiln_side_mask & (dot_ssr >= constants.SSR_FACING_COS_MIN)
      if not np.any(facing_mask):
        facing_mask = kiln_side_mask
      facing_areas = mesh.areas[facing_mask]
      t_kiln_facing_k = float(
          np.sum(t_safe[facing_mask] * facing_areas) / np.sum(facing_areas)
      )
    q_bot_conv_w = (
        float(
            np.sum(
                np.maximum(0.0, q_conv_out[kiln_bot_mask])
                * mesh.areas[kiln_bot_mask]
            )
        )
        if np.any(kiln_bot_mask)
        else 0.0
    )
    t_ssr_air_k, t_ssr_case_k = _solve_ssr_internal_air(
        t_ssr_faces_k=t_safe[ssr_mask],
        ssr_areas=mesh.areas[ssr_mask],
        t_ref_k=t_ref_val,
        t_lower_k=t_room_lower_k,
        params=params,
        t_kiln_facing_k=t_kiln_facing_k,
        q_bot_conv_w=q_bot_conv_w,
    )
    t_back_k[ssr_mask] = t_ssr_air_k

    # 5. Assemble conductive/source heat flux q_cond_in
    q_cond_in = np.zeros(n_patches, dtype=np.float64)
    q_cond_in[kiln_side_mask] = params.u_kiln_side * (
        params.t_internal_k - t_safe[kiln_side_mask]
    )
    q_cond_in[kiln_lid_mask] = params.u_kiln_top * (
        params.t_internal_k - t_safe[kiln_lid_mask]
    )
    q_cond_in[kiln_bot_mask] = params.u_kiln_bot * (
        params.t_internal_k - t_safe[kiln_bot_mask]
    )
    q_cond_in[shield_mask] = -chimney_sol.q_back_from_shield_w_m2[shield_mask]
    q_cond_in[gypsum_mask] = -u_gypsum_wall * (
        t_safe[gypsum_mask] - params.t_ext_k
    )
    t_back_k[gypsum_mask] = t_safe[gypsum_mask] + (
        q_cond_in[gypsum_mask] * (params.t_gypsum / params.k_gypsum)
    )
    q_under_ref_conv_w = 0.0
    if np.any(ref_mask):
      q_ref_down, du_ref_down, t_ref_back_k, t_pad_center_k = (
          solve_under_reflector_column(
              t_ref_front_k=t_ref_val,
              t_room_lower_k=t_room_lower_k,
              params=params,
          )
      )
      u_cond_diag[ref_mask] = du_ref_down
      q_cond_in[ref_mask] = -q_ref_down
      t_back_k[ref_mask] = t_ref_back_k
      if (
          params.r_reflector > 0.0
          and params.g_floor_reflector > constants.GEOM_TOL_M
      ):
        q_under_ref_conv_w = (
            float(np.sum(mesh.areas[ref_mask]))
            * constants.H_CONV_UNDER_REFLECTOR_W_M2K
            * (
                (t_ref_back_k - t_room_lower_k)
                + (t_pad_center_k - t_room_lower_k)
            )
        )
    q_cond_in[ssr_mask] = (
        constants.H_CONV_SSR_INTERNAL_W_M2K * (t_ssr_air_k - t_safe[ssr_mask])
    )
    q_ssr_louver_w = params.q_ssr_internal_w - (
        float(np.sum(mesh.areas[ssr_mask] * q_cond_in[ssr_mask]))
        if np.any(ssr_mask)
        else 0.0
    )

    residual = q_rad_out + q_conv_out - q_cond_in

    jac_diag = (
        4.0 * eps * params_lib.STEFAN_BOLTZMANN * (t_safe ** 3)
        + 1.25 * h_front
        + u_cond_diag
    )

    latest_state["t_front_k"] = t_safe
    latest_state["t_back_k"] = t_back_k.copy()
    latest_state["radiosity"] = radiosity
    latest_state["q_rad_out"] = q_rad_out
    latest_state["q_conv_out"] = q_conv_out
    latest_state["q_cond_in"] = q_cond_in
    latest_state["chimney"] = chimney_sol
    latest_state["t_ssr_air_k"] = t_ssr_air_k
    latest_state["t_ssr_case_k"] = t_ssr_case_k
    latest_state["q_rad_room_patch_w"] = q_rad_room_patch_w
    latest_state["q_ssr_louver_w"] = q_ssr_louver_w
    latest_state["q_under_ref_conv_w"] = q_under_ref_conv_w

    return residual, jac_diag

  room_sol: plume_room_lib.RoomSolution | None = None
  converged = False
  max_res = 1e9
  dt_room_change = 0.0
  dz_layer_change = 0.0

  def _run_inner_newton() -> None:
    nonlocal t_front_k, max_res
    for inner_iter in range(constants.SOLVER_MAX_INNER_ITERS):
      do_chimney = (inner_iter < 12) or (inner_iter % 2 == 0) or (max_res < 5.0)
      res_vec, jac_diag = _evaluate_step(t_front_k, update_chimney=do_chimney)
      max_res = float(np.max(np.abs(res_vec)))
      if max_res < constants.SOLVER_INNER_RES_TOL_W_M2 and do_chimney:
        break

      delta_t = -res_vec / np.maximum(jac_diag, 1.0)
      max_step = float(np.max(np.abs(delta_t)))
      damping = min(
          1.0, constants.SOLVER_MAX_STEP_K / max(max_step, 1e-9)
      )
      t_front_k = np.clip(
          t_front_k + damping * delta_t, constants.SOLVER_MIN_TEMP_K, t_max_bound
      )

    res_vec, _ = _evaluate_step(t_front_k, update_chimney=True)
    max_res = float(np.max(np.abs(res_vec)))

  def _solve_room() -> plume_room_lib.RoomSolution:
    return plume_room_lib.solve_plume_and_room(
        mesh=mesh,
        t_front_k=t_front_k,
        q_conv_front_w_m2=latest_state["q_conv_out"],
        chimney=latest_state["chimney"],
        t_lower_prev_k=t_room_lower_k,
        params=params,
        q_rad_room_patch_w=latest_state["q_rad_room_patch_w"],
        q_ssr_louver_w=latest_state["q_ssr_louver_w"],
        q_under_ref_conv_w=latest_state["q_under_ref_conv_w"],
    )

  for outer_iter in range(constants.SOLVER_MAX_OUTER_ITERS):
    _run_inner_newton()
    room_sol = _solve_room()

    dt_room_change = max(
        abs(room_sol.t_room_upper_k - t_room_upper_k),
        abs(room_sol.t_room_lower_k - t_room_lower_k),
    )
    dz_layer_change = abs(room_sol.z_stratification_layer_m - z_layer_m)
    t_room_upper_k = 0.7 * room_sol.t_room_upper_k + 0.3 * t_room_upper_k
    t_room_lower_k = 0.7 * room_sol.t_room_lower_k + 0.3 * t_room_lower_k
    z_layer_m = room_sol.z_stratification_layer_m

    logging.info(
        "Outer iter %d: max_res=%.3e W/m^2, T_upper=%.2f C, T_lower=%.2f C, "
        "dt_room=%.3f K, dz_layer=%.4f m",
        outer_iter + 1,
        max_res,
        t_room_upper_k - params_lib.KELVIN_OFFSET,
        t_room_lower_k - params_lib.KELVIN_OFFSET,
        dt_room_change,
        dz_layer_change,
    )
    if (
        max_res < constants.SOLVER_OUTER_RES_TOL_W_M2
        and dt_room_change < constants.SOLVER_OUTER_TEMP_TOL_K
        and dz_layer_change < constants.SOLVER_OUTER_LAYER_TOL_M
        and (
            outer_iter >= 1
            or (
                dt_room_change < constants.SOLVER_OUTER_FAST_TEMP_TOL_K
                and dz_layer_change < 0.5 * constants.SOLVER_OUTER_LAYER_TOL_M
            )
        )
    ):
      converged = True
      break

  assert room_sol is not None
  # Final patch polish synchronized directly to the undamped room solution.
  t_room_upper_k = room_sol.t_room_upper_k
  t_room_lower_k = room_sol.t_room_lower_k
  z_layer_m = room_sol.z_stratification_layer_m
  _run_inner_newton()
  room_sol = _solve_room()
  post_shift = max(
      abs(room_sol.t_room_upper_k - t_room_upper_k),
      abs(room_sol.t_room_lower_k - t_room_lower_k),
      abs(room_sol.z_stratification_layer_m - z_layer_m),
  )
  if post_shift > 1e-4:
    t_room_upper_k = room_sol.t_room_upper_k
    t_room_lower_k = room_sol.t_room_lower_k
    z_layer_m = room_sol.z_stratification_layer_m
    _run_inner_newton()
    room_sol = _solve_room()
  t_room_upper_k = room_sol.t_room_upper_k
  t_room_lower_k = room_sol.t_room_lower_k
  z_layer_m = room_sol.z_stratification_layer_m
  res_vec, _ = _evaluate_step(t_front_k, update_chimney=True)
  max_res = float(np.max(np.abs(res_vec)))

  wall_time_s = time.monotonic() - t_start

  return SolverSolution(
      t_front_k=latest_state["t_front_k"],
      t_back_k=latest_state["t_back_k"],
      radiosity_w_m2=latest_state["radiosity"],
      q_rad_net_w_m2=latest_state["q_rad_out"],
      q_conv_net_w_m2=latest_state["q_conv_out"],
      q_cond_net_w_m2=latest_state["q_cond_in"],
      chimney=latest_state["chimney"],
      room=room_sol,
      t_ssr_air_k=float(latest_state["t_ssr_air_k"]),
      t_ssr_case_k=float(latest_state["t_ssr_case_k"]),
      converged=converged,
      max_energy_residual_w_m2=max_res,
      solver_wall_time_s=wall_time_s,
      dt_room_change_k=float(dt_room_change),
  )