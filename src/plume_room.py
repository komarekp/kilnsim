"""Corner-modified MTT turbulent plume, hood capture, and 2-zone room model."""

import dataclasses
import math
import numpy as np
from src import chimney as chimney_lib
from src import mesh as mesh_lib
from src import params as params_lib


@dataclasses.dataclass(frozen=True)
class RoomSolution:
  """Resolved state of the rising thermal plume, hood capture, and room zones.

  Attributes:
    q_plume_conv_w: Total convective heat rate feeding the primary kiln plume (W).
    v_dot_plume_at_hood_m3_s: Volumetric flow rate of the primary plume at the
      hood inlet elevation z_hood_inlet (m^3/s).
    r_plume_at_hood_m: Characteristic 95% Gaussian capture radius of the primary
      plume at z_hood_inlet (m).
    m_eff_corner: Effective Hasemi/Mowrer corner image multiplier at the hood
      elevation (dimensionless, in [1.0, 4.0]).
    plume_fully_captured: True if the hood fan volumetric flow exceeds plume
      flow AND the hood radius encloses the plume capture radius.
    plume_capture_fraction: Fraction of primary plume enthalpy captured by the
      overhead hood (0.0 to 1.0).
    chimney_capture_fraction: Fraction of behind-shield chimney outlet flow
      drawn into the overhead hood (0.0 to 1.0).
    t_room_upper_k: Steady-state temperature of the upper room gas layer (K).
    t_room_lower_k: Steady-state temperature of the lower make-up air layer (K).
    z_stratification_layer_m: Elevation of the two-zone interface above z=0 (m).
  """

  q_plume_conv_w: float
  v_dot_plume_at_hood_m3_s: float
  r_plume_at_hood_m: float
  m_eff_corner: float
  plume_fully_captured: bool
  plume_capture_fraction: float
  chimney_capture_fraction: float
  t_room_upper_k: float
  t_room_lower_k: float
  z_stratification_layer_m: float


def solve_plume_and_room(
    mesh: mesh_lib.SurfaceMesh,
    t_front_k: np.ndarray,
    q_conv_front_w_m2: np.ndarray,
    chimney: chimney_lib.ChimneySolution,
    t_lower_prev_k: float,
    params: params_lib.SimParams,
) -> RoomSolution:
  """Solves the corner-modified MTT plume, hood capture, and 2-zone room model.

  Args:
    mesh: Discretized SurfaceMesh instance.
    t_front_k: (N_patches,) array of current front-face temperatures (K).
    q_conv_front_w_m2: (N_patches,) array of front convective fluxes (W/m^2).
    chimney: Solved ChimneySolution for the current iteration.
    t_lower_prev_k: Current lower-zone air temperature estimate (K).
    params: Validated SimParams instance.

  Returns:
    Populated RoomSolution instance.
  """
  stypes = mesh.surface_types
  areas = mesh.areas

  # 1. Sum convective heat release feeding the primary kiln plume (kiln side
  # boundary layer + kiln lid top + open exhaust bung if not locally captured).
  kiln_mask = (stypes == int(mesh_lib.SurfaceType.KILN_SIDE)) | (
      stypes == int(mesh_lib.SurfaceType.KILN_LID)
  )
  q_kiln_conv_w = float(
      np.sum(np.maximum(0.0, q_conv_front_w_m2[kiln_mask]) * areas[kiln_mask])
  )
  q_bung_w = 0.0 if params.bung_captured else params.q_bung_open_w
  q_plume_total_w = max(0.0, q_kiln_conv_w + q_bung_w)

  # Handle the unheated / kiln-turned-off limit cleanly:
  if q_plume_total_w <= 1e-3:
    return RoomSolution(
        q_plume_conv_w=0.0,
        v_dot_plume_at_hood_m3_s=0.0,
        r_plume_at_hood_m=0.0,
        m_eff_corner=1.0,
        plume_fully_captured=True,
        plume_capture_fraction=1.0,
        chimney_capture_fraction=1.0,
        t_room_upper_k=params.t_ext_k,
        t_room_lower_k=params.t_ext_k,
        z_stratification_layer_m=params.z_hood_inlet,
    )

  # 2. Virtual origin z_v of an area source of diameter d_kiln (Heskestad / MTT)
  q_kw = q_plume_total_w * 1e-3
  z_v = (
      params.z_kiln_top
      - 1.02 * params.d_kiln
      + 0.083 * (q_kw ** (2.0 / 5.0))
  )
  dz_plume = max(0.15, params.z_hood_inlet - z_v)

  # 3. Hasemi / Mowrer-Williamson corner standoff image multiplier m_eff(z).
  s_eff = min(params.s_kiln_left, params.s_kiln_back) + params.r_kiln
  b_gaussian = 0.12 * dz_plume
  m_eff = 1.0 + 3.0 * math.exp(-((s_eff / max(b_gaussian, 1e-3)) ** 2))

  # 4. Corner-modified MTT volumetric plume flow rate at z = z_hood_inlet.
  m_dot_plume_kg_s = (
      0.071
      * ((q_kw * m_eff) ** (1.0 / 3.0))
      * (dz_plume ** (5.0 / 3.0))
      / (m_eff ** (1.0 / 3.0))
  )
  m_dot_plume_kg_s += 0.018 * (q_kw ** (1.0 / 3.0)) * params.d_kiln
  dt_plume_mean = q_plume_total_w / (
      max(m_dot_plume_kg_s, 1e-4) * chimney_lib.AIR_CP
  )
  rho_plume = chimney_lib.AIR_RHO_REF * (
      t_lower_prev_k / (t_lower_prev_k + dt_plume_mean)
  )
  v_dot_plume_m3_s = m_dot_plume_kg_s / max(rho_plume, 0.5)

  # 5. Geometric capture radius at the hood mouth.
  hood_offset = math.hypot(
      params.x_hood - params.x_kiln_center,
      params.y_hood - params.y_kiln_center,
  )
  r_plume_95 = 1.65 * b_gaussian + hood_offset
  if not params.bung_captured:
    r_bung_overhang = max(0.0, params.r_bung - params.r_kiln)
    r_plume_95 += 0.35 * r_bung_overhang

  if params.r_hood_inlet >= r_plume_95:
    eta_geom = 1.0
  else:
    sigma_r = max(0.05, r_plume_95 / 2.45)
    eta_geom = 1.0 - math.exp(
        -0.5 * ((params.r_hood_inlet / sigma_r) ** 2)
    )

  if params.v_dot_hood_m3_s <= 0.0:
    eta_vol = 0.0
  else:
    eta_vol = min(1.0, params.v_dot_hood_m3_s / max(v_dot_plume_m3_s, 1e-5))

  eta_plume = eta_geom * eta_vol
  plume_fully_captured = (
      (params.v_dot_hood_m3_s >= v_dot_plume_m3_s)
      and (params.r_hood_inlet >= r_plume_95)
  )

  # 6. Behind-shield chimney capture fraction.
  v_dot_surplus = max(0.0, params.v_dot_hood_m3_s - eta_plume * v_dot_plume_m3_s)
  v_dot_chimney = chimney.m_dot_total_kg_s / chimney_lib.AIR_RHO_REF
  z_shield_top = params.h_wall - params.g_ceil
  dist_wall_to_hood_rim = max(
      0.01,
      min(params.x_hood, params.y_hood)
      - params.x_shield_front_back_wall
      - params.r_hood_inlet,
  )

  if v_dot_surplus <= 0.0 or v_dot_chimney <= 1e-5:
    eta_chimney = 0.0
  else:
    r_sink = math.hypot(
        dist_wall_to_hood_rim, z_shield_top - params.z_hood_inlet
    )
    v_sink = v_dot_surplus / (2.0 * math.pi * max(r_sink * r_sink, 0.04))
    v_chim_mean = 0.5 * (
        float(np.mean(chimney.v_gap_left)) + float(np.mean(chimney.v_gap_back))
    )
    velocity_capture_ratio = min(1.0, v_sink / max(v_chim_mean, 0.05))
    vol_capture_ratio = min(1.0, v_dot_surplus / v_dot_chimney)
    elevation_factor = (
        1.0
        if z_shield_top <= params.z_hood_inlet + 0.05
        else math.exp(-4.0 * (z_shield_top - params.z_hood_inlet))
    )
    eta_chimney = (
        velocity_capture_ratio * vol_capture_ratio * elevation_factor
    )

  # 7. Two-Zone Room Energy & Doorway Buoyancy Exchange Balance.
  shield_gyp_mask = (
      (stypes == int(mesh_lib.SurfaceType.SHIELD_LEFT))
      | (stypes == int(mesh_lib.SurfaceType.SHIELD_BACK))
      | (stypes == int(mesh_lib.SurfaceType.GYPSUM_LEFT))
      | (stypes == int(mesh_lib.SurfaceType.GYPSUM_BACK))
  )
  q_wall_front_conv_w = float(
      np.sum(
          np.maximum(0.0, q_conv_front_w_m2[shield_gyp_mask])
          * areas[shield_gyp_mask]
      )
  )

  q_spill_upper_w = (
      (1.0 - eta_plume) * q_plume_total_w
      + (1.0 - eta_chimney) * chimney.q_conv_total_w
      + 0.65 * q_wall_front_conv_w
  )

  m_dot_spill = (
      (1.0 - eta_plume) * m_dot_plume_kg_s
      + (1.0 - eta_chimney) * chimney.m_dot_total_kg_s
  )
  a_door = max(0.01, params.w_door_open * params.h_door)
  if m_dot_spill <= 1e-4:
    z_layer = max(params.z_hood_inlet, 0.85 * params.h_wall)
  else:
    z_layer = max(
        0.55 * params.h_door,
        min(
            params.z_hood_inlet,
            params.h_door / (1.0 + 1.5 * (m_dot_spill / max(a_door, 0.05))),
        ),
    )

  ua_upper = params.compute_ua_upper_zone(z_layer)
  ua_lower = params.compute_ua_lower_zone(z_layer)
  q_upper_heating_w = 0.92 * q_spill_upper_w
  q_lower_heating_w = 0.35 * q_wall_front_conv_w + 0.08 * q_spill_upper_w

  dt_upper = 0.5
  dt_lower = 0.2
  for _ in range(12):
    h_upper_door = max(0.05, params.h_door - z_layer)
    v_door_buoy = 0.65 * math.sqrt(
        params_lib.GRAVITY * h_upper_door * (max(0.0, dt_upper) / params.t_ext_k)
    )
    m_dot_door_buoy = (
        0.5
        * chimney_lib.AIR_RHO_REF
        * params.w_door_open
        * h_upper_door
        * v_door_buoy
    )
    m_dot_hood_flush = 0.25 * chimney_lib.AIR_RHO_REF * params.v_dot_hood_m3_s
    g_upper_ext = (
        ua_upper
        + (m_dot_door_buoy + m_dot_hood_flush) * chimney_lib.AIR_CP
    )

    m_dot_lower_ext = (
        chimney_lib.AIR_RHO_REF * params.v_dot_hood_m3_s + m_dot_door_buoy
    )
    g_lower_ext = ua_lower + m_dot_lower_ext * chimney_lib.AIR_CP

    m_dot_recirc = max(0.0, m_dot_spill - m_dot_door_buoy)
    g_recirc = m_dot_recirc * chimney_lib.AIR_CP

    denom = (
        g_upper_ext * g_lower_ext
        + g_recirc * (g_upper_ext + g_lower_ext)
    )
    if denom > 1e-6:
      dt_upper_new = (
          q_upper_heating_w * (g_lower_ext + g_recirc)
          + q_lower_heating_w * g_recirc
      ) / denom
      dt_lower_new = (
          q_lower_heating_w * (g_upper_ext + g_recirc)
          + q_upper_heating_w * g_recirc
      ) / denom
    else:
      dt_upper_new = q_upper_heating_w / max(g_upper_ext, 1.0)
      dt_lower_new = dt_upper_new

    dt_upper = 0.5 * dt_upper + 0.5 * max(0.0, dt_upper_new)
    dt_lower = 0.5 * dt_lower + 0.5 * max(0.0, min(dt_upper, dt_lower_new))

  t_room_upper_k = params.t_ext_k + dt_upper
  t_room_lower_k = params.t_ext_k + dt_lower

  return RoomSolution(
      q_plume_conv_w=q_plume_total_w,
      v_dot_plume_at_hood_m3_s=v_dot_plume_m3_s,
      r_plume_at_hood_m=r_plume_95,
      m_eff_corner=m_eff,
      plume_fully_captured=plume_fully_captured,
      plume_capture_fraction=eta_plume,
      chimney_capture_fraction=eta_chimney,
      t_room_upper_k=t_room_upper_k,
      t_room_lower_k=t_room_lower_k,
      z_stratification_layer_m=z_layer,
  )