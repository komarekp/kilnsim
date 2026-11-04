"""1D vertical buoyant channel solver for the behind-shield chimney air gap.

Solves the coupled hydrostatic buoyancy draft, Darcy-Weisbach and orifice pressure
losses, convective air heating dT_gap/dz, across-gap radiation, and steady 1D
conduction through the structural AAC block wall for each vertical shield column.
"""

import dataclasses
import numpy as np
from src import mesh as mesh_lib
from src import params as params_lib

# Dry air thermophysical properties at ~315 K and 1 atm.
AIR_RHO_REF: float = 1.12  # kg / m^3
AIR_CP: float = 1007.0  # J / (kg * K)
AIR_MU: float = 1.91e-5  # Pa * s
AIR_NU: float = 1.70e-5  # m^2 / s
AIR_K: float = 0.0274  # W / (m * K)
AIR_PR: float = 0.705  # Prandtl number


@dataclasses.dataclass(frozen=True)
class ChimneySolution:
  """Resolved thermal and fluid state of the behind-shield chimney channels.

  Attributes:
    v_gap_left: (n_cols_left,) upward air velocity in each left-wall column (m/s).
    v_gap_back: (n_cols_back,) upward air velocity in each back-wall column (m/s).
    t_exit_left_k: (n_cols_left,) exit air temperature at top of left shield (K).
    t_exit_back_k: (n_cols_back,) exit air temperature at top of back shield (K).
    t_air_patch_k: (N_patches,) local behind-shield air temperature for each
      mesh patch (set to t_inlet_k for non-shield patches) in Kelvin.
    t_aac_front_patch_k: (N_patches,) solved AAC block front-face temperature
      behind each shield patch in Kelvin.
    h_chan_patch: (N_patches,) internal channel convective heat transfer
      coefficient for each shield patch in W/(m^2*K).
    h_eff_back_patch: (N_patches,) effective linearized derivative
      `dq_back / dT_shield_back` (W/(m^2*K)) for fast Newton convergence.
    q_back_from_shield_w_m2: (N_patches,) total heat flux leaving the back face
      of each shield patch (convection to gap air + radiation to AAC wall, or
      direct solid conduction when d_chimney == 0) in W/m^2.
    m_dot_total_kg_s: Total buoyant mass flow rate exiting both shield chimneys
      in kg/s.
    q_conv_total_w: Total convective enthalpy rate carried out the top of both
      shield chimneys relative to t_inlet_k in Watts.
  """

  v_gap_left: np.ndarray
  v_gap_back: np.ndarray
  t_exit_left_k: np.ndarray
  t_exit_back_k: np.ndarray
  t_air_patch_k: np.ndarray
  t_aac_front_patch_k: np.ndarray
  h_chan_patch: np.ndarray
  h_eff_back_patch: np.ndarray
  q_back_from_shield_w_m2: np.ndarray
  m_dot_total_kg_s: float
  q_conv_total_w: float


def _solve_wall_columns(
    t_shield_back_grid: np.ndarray,
    dz: float,
    t_inlet_k: float,
    params: params_lib.SimParams,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
  """Vectorized 1D buoyant flow and AAC wall solver across all columns at once.

  Handles flush solid contact (`d_chimney_eff == 0`), stagnant closed cavities
  (`g_floor == 0` or `g_ceil == 0`), and open buoyant channels (`d_chimney > 0`,
  `g_floor > 0`, `g_ceil > 0`).

  Args:
    t_shield_back_grid: (n_cols, n_rows) array of shield back-face temperatures
      in Kelvin.
    dz: Vertical height of each row patch in metres.
    t_inlet_k: Ambient air temperature entering at z = g_floor in Kelvin.
    params: Validated SimParams instance.

  Returns:
    Tuple of (v_cols, t_exit_cols, t_air_grid, t_aac_grid, h_chan_grid,
    h_eff_back_grid, q_back_grid).
  """
  n_cols, n_rows = t_shield_back_grid.shape
  u_aac_cond = 1.0 / ((params.t_aac / params.k_aac) + params_lib.R_SE_WALL)
  d_eff = params.d_chimney_eff

  # Case 1: Flush-mounted shield with zero chimney gap (direct solid conduction
  # from shield back face into AAC block wall).
  if d_eff <= 1e-6:
    v_cols = np.zeros(n_cols, dtype=np.float64)
    t_exit_cols = np.full(n_cols, t_inlet_k, dtype=np.float64)
    t_air_grid = t_shield_back_grid.copy()
    t_aac_grid = t_shield_back_grid.copy()
    h_chan_grid = np.zeros((n_cols, n_rows), dtype=np.float64)
    h_eff_back_grid = np.full((n_cols, n_rows), u_aac_cond, dtype=np.float64)
    q_back_grid = u_aac_cond * (t_shield_back_grid - params.t_ext_k)
    return (
        v_cols,
        t_exit_cols,
        t_air_grid,
        t_aac_grid,
        h_chan_grid,
        h_eff_back_grid,
        q_back_grid,
    )

  # Case 2: Finite air gap d_eff > 0 (either open chimney or closed cavity).
  d_hyd = 2.0 * d_eff
  h_shield = params.h_shield
  eps_eff_gap = 1.0 / (
      (1.0 / params.eps_shield) + (1.0 / params.eps_aac) - 1.0
  )

  # Determine which columns have an open through-flow path.
  open_flow_mask = np.ones(n_cols, dtype=bool)
  if params.g_ceil <= 1e-6:
    open_flow_mask[:] = False
  elif params.g_floor <= 1e-6:
    open_flow_mask[:] = False
    if not params.seal_chimney_sides and n_cols > 0:
      open_flow_mask[-1] = True

  ratio_in = d_eff / max(params.g_floor, 1e-6)
  ratio_out = d_eff / max(params.g_ceil, 1e-6)
  k_in_base = 1.5 * (1.0 + ratio_in * ratio_in)
  k_out = 1.0 * (1.0 + ratio_out * ratio_out)

  k_in = np.full(n_cols, k_in_base, dtype=np.float64)
  side_leak_rate = np.zeros(n_cols, dtype=np.float64)
  if not params.seal_chimney_sides and n_cols > 0:
    k_in[-1] = 2.5 if params.g_floor <= 1e-6 else 0.65 * k_in_base
    side_leak_rate[-1] = 0.15

  leak_step = side_leak_rate * (dz / h_shield)
  keep_step = 1.0 - leak_step

  v_guess = np.where(open_flow_mask, 0.25, 0.0)
  t_aac_grid = np.full((n_cols, n_rows), t_inlet_k, dtype=np.float64)
  t_air_grid = np.full((n_cols, n_rows), t_inlet_k, dtype=np.float64)
  t_exit_cols = np.full(n_cols, t_inlet_k, dtype=np.float64)

  t_mean_shield = np.mean(t_shield_back_grid, axis=1)
  ra_gap = (
      params_lib.GRAVITY
      * np.maximum(0.0, t_mean_shield - t_inlet_k)
      * (d_eff ** 3)
      / (t_inlet_k * AIR_NU * (AIR_NU / AIR_PR))
  )
  nu_nat = 0.22 * (ra_gap ** 0.28)

  for _ in range(6):
    re_d = v_guess * d_hyd / AIR_NU
    gz = (d_hyd / max(h_shield, 0.1)) * re_d * AIR_PR
    nu_forced = 7.54 + (0.03 * gz) / (1.0 + 0.016 * (gz ** (2.0 / 3.0)))
    nu_chan = np.sqrt(nu_forced * nu_forced + nu_nat * nu_nat)
    h_chan = np.maximum(3.5, (nu_chan * AIR_K) / d_hyd)

    t_rad_mean = 0.5 * (t_shield_back_grid + t_aac_grid)
    h_rad_gap = (
        4.0
        * eps_eff_gap
        * params_lib.STEFAN_BOLTZMANN
        * (t_rad_mean ** 3)
    )

    m_dot_cp = AIR_RHO_REF * d_eff * v_guess * AIR_CP
    # Exact exponential NTU integration factor phi in [0, 1] over step dz:
    # As v_guess -> 0, ntu -> inf, exp_ntu -> 0, and phi -> 0 (stagnant cavity).
    ntu = np.where(
        m_dot_cp > 1e-9,
        (2.0 * h_chan * dz) / np.maximum(m_dot_cp, 1e-9),
        1.0e9,
    )
    exp_ntu = np.where(ntu < 50.0, np.exp(-np.minimum(ntu, 50.0)), 0.0)
    phi = np.where(
        ntu > 1e-5,
        (1.0 - exp_ntu) / ntu,
        1.0 - 0.5 * ntu,
    )
    h_adv = phi * h_chan

    t_air_curr = np.full(n_cols, t_inlet_k, dtype=np.float64)
    for j in range(n_rows):
      ts_b = t_shield_back_grid[:, j]
      hr = h_rad_gap[:, j]
      h_cross = hr + 0.5 * (1.0 - phi) * h_chan
      denom_aac = h_cross + h_adv + u_aac_cond
      t_aac_j = (
          h_cross * ts_b + h_adv * t_air_curr + u_aac_cond * params.t_ext_k
      ) / denom_aac
      t_aac_grid[:, j] = t_aac_j

      t_wall_mid = 0.5 * (ts_b + t_aac_j)
      t_air_mean_j = (1.0 - phi) * t_wall_mid + phi * t_air_curr
      t_air_out_j = t_wall_mid + (t_air_curr - t_wall_mid) * exp_ntu
      t_air_out_j = np.where(
          open_flow_mask,
          keep_step * t_air_out_j + leak_step * t_inlet_k,
          t_wall_mid,
      )
      t_air_grid[:, j] = t_air_mean_j
      t_air_curr = t_air_out_j

    t_exit_cols = np.where(open_flow_mask, t_air_curr, t_inlet_k)
    t_air_mean = np.mean(t_air_grid, axis=1)

    delta_p_buoy = (
        AIR_RHO_REF
        * params_lib.GRAVITY
        * h_shield
        * np.maximum(0.0, 1.0 - (t_inlet_k / np.maximum(t_air_mean, t_inlet_k)))
    )
    f_darcy = np.maximum(0.03, 96.0 / np.maximum(re_d, 1e-3))
    k_total = k_in + f_darcy * (h_shield / d_hyd) + k_out
    v_new = np.where(
        open_flow_mask,
        np.sqrt((2.0 * delta_p_buoy) / (AIR_RHO_REF * k_total)),
        0.0,
    )
    v_guess = 0.5 * v_guess + 0.5 * v_new

  h_chan_grid = np.broadcast_to(h_chan[:, None], (n_cols, n_rows)).copy()
  q_rad_exact = eps_eff_gap * params_lib.STEFAN_BOLTZMANN * (
      (t_shield_back_grid ** 4) - (t_aac_grid ** 4)
  )
  q_conv_back = h_chan[:, None] * (t_shield_back_grid - t_air_grid)
  q_back_grid = q_conv_back + q_rad_exact

  # Effective linearized derivative dq_back / dT_shield_back accounting for AAC
  # wall response (used by solver.py for 1-step back-surface convergence).
  h_cross_grid = h_rad_gap + 0.5 * (1.0 - phi[:, None]) * h_chan[:, None]
  h_adv_grid = h_adv[:, None]
  h_eff_back_grid = h_adv_grid + (
      h_cross_grid
      * (h_adv_grid + u_aac_cond)
      / (h_cross_grid + h_adv_grid + u_aac_cond)
  )

  return (
      v_guess,
      t_exit_cols,
      t_air_grid,
      t_aac_grid,
      h_chan_grid,
      h_eff_back_grid,
      q_back_grid,
  )


def solve_chimney(
    t_shield_back_k: np.ndarray,
    t_inlet_k: float,
    mesh: mesh_lib.SurfaceMesh,
    params: params_lib.SimParams,
) -> ChimneySolution:
  """Solves all behind-shield vertical chimney columns for both corner walls.

  Args:
    t_shield_back_k: (N_patches,) array of back-face temperatures in Kelvin.
    t_inlet_k: Lower-zone ambient air temperature entering at z = g_floor (K).
    mesh: Discretized SurfaceMesh instance.
    params: Validated SimParams instance.

  Returns:
    Populated ChimneySolution instance.
  """
  n_rows = mesh.n_rows_shield
  dz = params.h_shield / n_rows

  left_mask = mesh.chimney_wall_id == 0
  back_mask = mesh.chimney_wall_id == 1

  t_left_grid = np.full(
      (mesh.n_cols_left, n_rows), t_inlet_k, dtype=np.float64
  )
  t_left_grid[
      mesh.chimney_col_idx[left_mask], mesh.chimney_row_idx[left_mask]
  ] = t_shield_back_k[left_mask]

  t_back_grid = np.full(
      (mesh.n_cols_back, n_rows), t_inlet_k, dtype=np.float64
  )
  t_back_grid[
      mesh.chimney_col_idx[back_mask], mesh.chimney_row_idx[back_mask]
  ] = t_shield_back_k[back_mask]

  dx_left = params.w_shield_left / mesh.n_cols_left
  dx_back = params.w_shield_back / mesh.n_cols_back

  (
      v_left,
      t_exit_left,
      t_air_left,
      t_aac_left,
      h_left,
      h_eff_left,
      q_back_left,
  ) = _solve_wall_columns(
      t_shield_back_grid=t_left_grid,
      dz=dz,
      t_inlet_k=t_inlet_k,
      params=params,
  )

  (
      v_back,
      t_exit_back,
      t_air_back,
      t_aac_back,
      h_back,
      h_eff_back,
      q_back_back,
  ) = _solve_wall_columns(
      t_shield_back_grid=t_back_grid,
      dz=dz,
      t_inlet_k=t_inlet_k,
      params=params,
  )

  t_air_patch = np.full(mesh.num_patches, t_inlet_k, dtype=np.float64)
  t_aac_patch = np.full(mesh.num_patches, t_inlet_k, dtype=np.float64)
  h_chan_patch = np.zeros(mesh.num_patches, dtype=np.float64)
  h_eff_patch = np.zeros(mesh.num_patches, dtype=np.float64)
  q_back_patch = np.zeros(mesh.num_patches, dtype=np.float64)

  l_cols = mesh.chimney_col_idx[left_mask]
  l_rows = mesh.chimney_row_idx[left_mask]
  t_air_patch[left_mask] = t_air_left[l_cols, l_rows]
  t_aac_patch[left_mask] = t_aac_left[l_cols, l_rows]
  h_chan_patch[left_mask] = h_left[l_cols, l_rows]
  h_eff_patch[left_mask] = h_eff_left[l_cols, l_rows]
  q_back_patch[left_mask] = q_back_left[l_cols, l_rows]

  b_cols = mesh.chimney_col_idx[back_mask]
  b_rows = mesh.chimney_row_idx[back_mask]
  t_air_patch[back_mask] = t_air_back[b_cols, b_rows]
  t_aac_patch[back_mask] = t_aac_back[b_cols, b_rows]
  h_chan_patch[back_mask] = h_back[b_cols, b_rows]
  h_eff_patch[back_mask] = h_eff_back[b_cols, b_rows]
  q_back_patch[back_mask] = q_back_back[b_cols, b_rows]

  m_dot_left = AIR_RHO_REF * params.d_chimney_eff * dx_left * v_left
  m_dot_back = AIR_RHO_REF * params.d_chimney_eff * dx_back * v_back
  m_dot_total = float(np.sum(m_dot_left) + np.sum(m_dot_back))

  q_conv_left = np.sum(m_dot_left * AIR_CP * (t_exit_left - t_inlet_k))
  q_conv_back = np.sum(m_dot_back * AIR_CP * (t_exit_back - t_inlet_k))
  q_conv_total = float(q_conv_left + q_conv_back)

  return ChimneySolution(
      v_gap_left=v_left,
      v_gap_back=v_back,
      t_exit_left_k=t_exit_left,
      t_exit_back_k=t_exit_back,
      t_air_patch_k=t_air_patch,
      t_aac_front_patch_k=t_aac_patch,
      h_chan_patch=h_chan_patch,
      h_eff_back_patch=h_eff_patch,
      q_back_from_shield_w_m2=q_back_patch,
      m_dot_total_kg_s=m_dot_total,
      q_conv_total_w=q_conv_total,
  )