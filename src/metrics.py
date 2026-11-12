"""Safety metrics evaluation, point-temperature queries, and output assembly."""

import math
import numpy as np
from scipy import optimize
from src import mesh as mesh_lib
from src import params as params_lib
from src import solver as solver_lib
from src import view_factors as vf_lib
from proto import kiln_sim_pb2

# Engineering pass/fail temperature thresholds in degrees Celsius.
LIMIT_GYPSUM_C: float = 40.0
LIMIT_CEILING_C: float = 40.0
LIMIT_SSR_AIR_C: float = 35.0
LIMIT_SSR_CASE_C: float = 45.0
LIMIT_VIP_C: float = 50.0
LIMIT_VINYL_C: float = 45.0

# Floor surface emissivities and under-reflector / ceiling mixing coefficients.
FLOOR_PAD_EMISSIVITY: float = 0.88
VINYL_FLOOR_EMISSIVITY: float = 0.90
H_CONV_UNDER_REFLECTOR_W_M2K: float = 3.0
CHIMNEY_CEILING_MIXING_FRACTION: float = 0.35


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


def _solve_floor_sensor_temp_k(
    vf_vec: np.ndarray,
    radiosity_w_m2: np.ndarray,
    eps_floor: float,
    u_subfloor: float,
    t_amb_k: float,
    t_ext_k: float,
) -> float:
  """Solves the local radiative-convective-conductive equilibrium at a point."""
  f_room = max(0.0, 1.0 - float(np.sum(vf_vec)))
  g_incident = float(np.dot(vf_vec, radiosity_w_m2)) + (
      f_room * params_lib.STEFAN_BOLTZMANN * (t_amb_k ** 4)
  )
  q_abs = eps_floor * g_incident

  def _res(t_k: float) -> float:
    q_emit = eps_floor * params_lib.STEFAN_BOLTZMANN * (t_k ** 4)
    h_conv = max(2.0, 1.5 * (abs(t_k - t_amb_k) ** 0.25))
    q_conv = h_conv * (t_k - t_amb_k)
    q_cond = u_subfloor * (t_k - t_ext_k)
    return q_abs - q_emit - q_conv - q_cond

  sol = optimize.root_scalar(
      _res, bracket=[t_ext_k - 5.0, t_amb_k + 250.0], method="brentq"
  )
  return float(sol.root)


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

  # 3. Floor Under Reflector (directly under the center of the kiln)
  ref_mask = stypes == int(mesh_lib.SurfaceType.REFLECTOR_TOP)
  r_vip = params.t_vip / params.k_vip
  r_screed = params.t_screed / params.k_screed
  r_pad = (
      params.t_floor_pad / params.k_pad if params.t_floor_pad > 0.0 else 0.0
  )
  r_sub_total = r_pad + r_screed + r_vip
  u_pad_subfloor = 1.0 / r_sub_total

  if params.r_reflector <= 0.0:
    t_pad_center_k = _solve_floor_sensor_temp_k(
        vf_vec=vf_sys.sensor_vf["pad_center"],
        radiosity_w_m2=solution.radiosity_w_m2,
        eps_floor=FLOOR_PAD_EMISSIVITY,
        u_subfloor=u_pad_subfloor,
        t_amb_k=solution.room.t_room_lower_k,
        t_ext_k=params.t_ext_k,
    )
  elif params.g_floor_reflector > 1e-4:
    t_ref_back_k = float(solution.t_back_k[ref_mask][0])
    eps_gap = 1.0 / (
        (1.0 / params.eps_shield) + (1.0 / FLOOR_PAD_EMISSIVITY) - 1.0
    )
    h_rad_under = (
        4.0
        * eps_gap
        * params_lib.STEFAN_BOLTZMANN
        * ((0.5 * (t_ref_back_k + solution.room.t_room_lower_k)) ** 3)
    )
    t_pad_center_k = (
        h_rad_under * t_ref_back_k
        + H_CONV_UNDER_REFLECTOR_W_M2K * solution.room.t_room_lower_k
        + u_pad_subfloor * params.t_ext_k
    ) / (h_rad_under + H_CONV_UNDER_REFLECTOR_W_M2K + u_pad_subfloor)
  else:
    t_pad_center_k = float(solution.t_back_k[ref_mask][0])

  t_floor_pad_under_reflector_c = t_pad_center_k - params_lib.KELVIN_OFFSET
  t_vip_under_ref_k = params.t_ext_k + (
      (t_pad_center_k - params.t_ext_k) * (r_vip / r_sub_total)
  )
  t_vip_under_reflector_c = t_vip_under_ref_k - params_lib.KELVIN_OFFSET
  floor_under_reflector_safe = t_vip_under_reflector_c < LIMIT_VIP_C

  # 4. Floor Outside Reflector
  u_vinyl_subfloor = 1.0 / (r_screed + r_vip)

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
  t_aac_max_c = float(
      np.max(solution.chimney.t_aac_front_patch_k[shield_mask])
      - params_lib.KELVIN_OFFSET
  )
  v_chimney_max = max(
      float(np.max(solution.chimney.v_gap_left)),
      float(np.max(solution.chimney.v_gap_back)),
  )
  t_chimney_exit_max_k = max(
      float(np.max(solution.chimney.t_exit_left_k)),
      float(np.max(solution.chimney.t_exit_back_k)),
  )
  t_chimney_exit_max_c = t_chimney_exit_max_k - params_lib.KELVIN_OFFSET

  # 6. Ceiling peak surface temperature at z = h_wall
  uncaptured_chimney_frac = 1.0 - solution.room.chimney_capture_fraction
  t_amb_ceil_chimney_k = (
      solution.room.t_room_upper_k
      + CHIMNEY_CEILING_MIXING_FRACTION
      * uncaptured_chimney_frac
      * max(0.0, t_chimney_exit_max_k - solution.room.t_room_upper_k)
  )
  t_ceil_hood_corner_k = _solve_floor_sensor_temp_k(
      vf_vec=vf_sys.sensor_vf["ceiling_hood_rim_corner"],
      radiosity_w_m2=solution.radiosity_w_m2,
      eps_floor=params.eps_gypsum,
      u_subfloor=params.u_roof,
      t_amb_k=solution.room.t_room_upper_k,
      t_ext_k=params.t_ext_k,
  )
  t_ceil_hood_room_k = _solve_floor_sensor_temp_k(
      vf_vec=vf_sys.sensor_vf["ceiling_hood_rim_room"],
      radiosity_w_m2=solution.radiosity_w_m2,
      eps_floor=params.eps_gypsum,
      u_subfloor=params.u_roof,
      t_amb_k=solution.room.t_room_upper_k,
      t_ext_k=params.t_ext_k,
  )
  t_ceil_chimney_k = _solve_floor_sensor_temp_k(
      vf_vec=vf_sys.sensor_vf["ceiling_chimney_corner"],
      radiosity_w_m2=solution.radiosity_w_m2,
      eps_floor=params.eps_gypsum,
      u_subfloor=params.u_roof,
      t_amb_k=t_amb_ceil_chimney_k,
      t_ext_k=params.t_ext_k,
  )
  t_ceiling_max_c = (
      max(t_ceil_hood_corner_k, t_ceil_hood_room_k, t_ceil_chimney_k)
      - params_lib.KELVIN_OFFSET
  )
  ceiling_safe = t_ceiling_max_c < LIMIT_CEILING_C

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
  patches_pb = mesh.to_patch_protos(solution=solution)

  output_pb = kiln_sim_pb2.SimulationOutput(
      provenance=provenance,
      derived=derived_pb,
      metrics=metrics_pb,
      patches=patches_pb,
  )
  output_pb.input_config_snapshot.CopyFrom(config_pb)
  return output_pb