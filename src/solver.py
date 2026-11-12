"""Coupled radiative, convective, conductive, and room-zone thermal solver."""

import dataclasses
import math
import time
from typing import Any
from absl import logging
import numpy as np
from scipy import linalg
from src import chimney as chimney_lib
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
  t_film = np.maximum(250.0, 0.5 * (t_surf_k + t_amb_k))
  beta = 1.0 / t_film
  delta_t = np.sqrt((t_surf_k - t_amb_k) ** 2 + 1e-6)

  nu = 1.57e-5 * ((t_film / 300.0) ** 1.7)
  alpha = nu / chimney_lib.AIR_PR
  k_air = 0.0262 * ((t_film / 300.0) ** 0.8)

  ra_l = (
      params_lib.GRAVITY
      * beta
      * delta_t
      * (char_lengths_m ** 3)
      / np.maximum(nu * alpha, 1e-14)
  )

  pr_term = (1.0 + (0.492 / chimney_lib.AIR_PR) ** (9.0 / 16.0)) ** (
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

  return np.maximum(2.0, h_c)


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
    lu_piv: Pre-factored LU decomposition of `I - (1 - eps) * F_eff`.
    mesh: Discretized SurfaceMesh instance.
    vf_sys: Precomputed ViewFactorSystem instance.

  Returns:
    Tuple of (radiosity J in W/m^2, net outgoing radiative flux q_rad_out in
    W/m^2 where positive means net radiation leaving the surface).
  """
  eps = mesh.emissivities
  eb_surf = params_lib.STEFAN_BOLTZMANN * (t_front_k ** 4)
  eb_room = params_lib.STEFAN_BOLTZMANN * (t_amb_rad_k ** 4)

  rhs = eps * eb_surf + (1.0 - eps) * vf_sys.f_room * eb_room
  radiosity = linalg.lu_solve(lu_piv, rhs)
  q_rad_out = (eps / np.maximum(1.0 - eps, 1e-6)) * (eb_surf - radiosity)
  return radiosity, q_rad_out


def _solve_ssr_internal_air(
    t_ssr_faces_k: np.ndarray,
    ssr_areas: np.ndarray,
    t_ref_k: float,
    t_lower_k: float,
    params: params_lib.SimParams,
) -> tuple[float, float]:
  """Computes the steady-state SSR internal air and average casing temperatures."""
  t_ssr_case_k = float(np.sum(t_ssr_faces_k * ssr_areas) / np.sum(ssr_areas))

  a_ref = math.pi * (params.r_reflector ** 2)
  h_ref = 3.5
  q_ref_conv = max(0.0, h_ref * a_ref * (t_ref_k - t_lower_k))
  m_dot_local_inflow = 0.035  # kg/s
  dt_floor_conv = q_ref_conv / (m_dot_local_inflow * chimney_lib.AIR_CP)
  t_ssr_inlet_k = t_lower_k + dt_floor_conv

  a_louver_eff = 0.0022  # m^2 effective free louver opening area
  h_int_box = 5.5  # W/(m^2*K) internal wall-to-air convection
  a_box_total = float(np.sum(ssr_areas))
  ua_box = h_int_box * a_box_total

  t_air_k = 0.5 * (t_ssr_inlet_k + t_ssr_case_k) + (
      params.q_ssr_internal_w / max(ua_box, 1.0)
  )
  for _ in range(8):
    dt_buoy = abs(t_air_k - t_ssr_inlet_k)
    v_louver = 0.55 * math.sqrt(
        2.0 * params_lib.GRAVITY * params.h_ssr * (dt_buoy / t_ssr_inlet_k)
    )
    m_dot_louver = max(
        1.2e-3, chimney_lib.AIR_RHO_REF * a_louver_eff * v_louver
    )
    mcp = m_dot_louver * chimney_lib.AIR_CP
    t_air_new = (
        params.q_ssr_internal_w + mcp * t_ssr_inlet_k + ua_box * t_ssr_case_k
    ) / (mcp + ua_box)
    t_air_k = 0.5 * t_air_k + 0.5 * t_air_new

  return t_air_k, t_ssr_case_k


def solve_thermal_system(
    mesh: mesh_lib.SurfaceMesh,
    vf_sys: vf_lib.ViewFactorSystem,
    params: params_lib.SimParams,
) -> SolverSolution:
  """Executes the coupled inner (surface) and middle (room) solver loops.

  Uses a vectorized diagonal-Newton solver with exact analytical local
  derivatives `dR_m/dT_m` and full radiosity/chimney updates at each step,
  avoiding O(N^2) finite-difference Jacobian evaluations.

  Args:
    mesh: Discretized SurfaceMesh instance.
    vf_sys: Precomputed ViewFactorSystem instance.
    params: Validated SimParams instance.

  Returns:
    Converged SolverSolution instance.
  """
  t_start = time.monotonic()
  n_patches = mesh.num_patches
  stypes = mesh.surface_types
  z_coords = mesh.centroids[:, 2]
  eps = mesh.emissivities

  rad_matrix = np.eye(n_patches, dtype=np.float64) - (
      (1.0 - eps)[:, None] * vf_sys.f_eff
  )
  lu_piv = linalg.lu_factor(rad_matrix)

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
  if params.g_floor_reflector > 1e-4:
    u_ref_down = 1.0 / (
        (params.t_reflector_calsil / params.k_shield) + (1.0 / 3.5)
    )
  else:
    u_ref_down = 1.0 / (
        (params.t_reflector_calsil / params.k_shield)
        + (params.t_floor_pad / max(params.k_pad, 0.1))
        + (params.t_screed / params.k_screed)
        + (params.t_vip / params.k_vip)
    )

  g_shield = params.k_shield / params.t_shield

  u_cond_diag = np.zeros(n_patches, dtype=np.float64)
  u_cond_diag[kiln_side_mask] = params.u_kiln_side
  u_cond_diag[kiln_lid_mask] = params.u_kiln_top
  u_cond_diag[kiln_bot_mask] = params.u_kiln_bot
  u_cond_diag[gypsum_mask] = u_gypsum_wall
  u_cond_diag[ref_mask] = u_ref_down
  u_cond_diag[ssr_mask] = 5.5

  dt_firing = max(0.0, params.t_internal_k - params.t_ext_k)
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

    # 4. SSR internal air & casing balance
    t_ref_val = (
        float(t_safe[ref_mask][0]) if np.any(ref_mask) else t_room_lower_k
    )
    t_ssr_air_k, t_ssr_case_k = _solve_ssr_internal_air(
        t_ssr_faces_k=t_safe[ssr_mask],
        ssr_areas=mesh.areas[ssr_mask],
        t_ref_k=t_ref_val,
        t_lower_k=t_room_lower_k,
        params=params,
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
    q_cond_in[ref_mask] = -u_ref_down * (t_safe[ref_mask] - t_room_lower_k)
    t_back_k[ref_mask] = t_safe[ref_mask] + (
        q_cond_in[ref_mask] * (params.t_reflector_calsil / params.k_shield)
    )
    q_cond_in[ssr_mask] = 5.5 * (t_ssr_air_k - t_safe[ssr_mask])

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

    return residual, jac_diag

  room_sol: plume_room_lib.RoomSolution | None = None
  converged = False
  max_res = 1e9
  dt_room_change = 0.0

  for outer_iter in range(20):
    for inner_iter in range(60):
      do_chimney = (inner_iter < 12) or (inner_iter % 2 == 0) or (max_res < 5.0)
      res_vec, jac_diag = _evaluate_step(t_front_k, update_chimney=do_chimney)
      max_res = float(np.max(np.abs(res_vec)))
      if max_res < 1e-4 and do_chimney:
        break

      delta_t = -res_vec / np.maximum(jac_diag, 1.0)
      max_step = float(np.max(np.abs(delta_t)))
      damping = min(1.0, 25.0 / max(max_step, 1e-9))
      t_front_k = np.clip(t_front_k + damping * delta_t, 250.0, t_max_bound)

    res_vec, _ = _evaluate_step(t_front_k, update_chimney=True)
    max_res = float(np.max(np.abs(res_vec)))

    room_sol = plume_room_lib.solve_plume_and_room(
        mesh=mesh,
        t_front_k=t_front_k,
        q_conv_front_w_m2=latest_state["q_conv_out"],
        chimney=latest_state["chimney"],
        t_lower_prev_k=t_room_lower_k,
        params=params,
    )

    dt_room_change = max(
        abs(room_sol.t_room_upper_k - t_room_upper_k),
        abs(room_sol.t_room_lower_k - t_room_lower_k),
    )
    t_room_upper_k = 0.7 * room_sol.t_room_upper_k + 0.3 * t_room_upper_k
    t_room_lower_k = 0.7 * room_sol.t_room_lower_k + 0.3 * t_room_lower_k
    z_layer_m = room_sol.z_stratification_layer_m

    logging.info(
        "Outer iter %d: max_res=%.3e W/m^2, T_upper=%.2f C, T_lower=%.2f C, "
        "dt_room=%.3f K",
        outer_iter + 1,
        max_res,
        t_room_upper_k - params_lib.KELVIN_OFFSET,
        t_room_lower_k - params_lib.KELVIN_OFFSET,
        dt_room_change,
    )
    if max_res < 0.5 and dt_room_change < 0.05:
      converged = True
      break

  wall_time_s = time.monotonic() - t_start
  assert room_sol is not None

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