"""SI-unit simulation parameters, derived geometry, and physical bounds validation.

Converts user-supplied millimetre, Celsius, and CFM parameters from
SimulationConfig into an immutable SI-unit (metres, Kelvin, Watts, m^3/s)
dataclass and computes all derived vertical datums and geometric relationships.
"""

import dataclasses
import math
from absl import logging
from proto import kiln_sim_pb2

# Physical constants in SI units.
KELVIN_OFFSET: float = 273.15
MM_TO_M: float = 1.0e-3
M_TO_MM: float = 1.0e3
CFM_TO_M3_S: float = 4.7194745e-4
STEFAN_BOLTZMANN: float = 5.670374419e-8  # W / (m^2 * K^4)
GRAVITY: float = 9.80665  # m / s^2

# Standard interior and exterior film resistances for unshielded wall U-value.
R_SI_WALL: float = 0.13  # m^2 * K / W
R_SE_WALL: float = 0.04  # m^2 * K / W

# Symbolic validation bounds in SI units (metres and dimensionless).
MIN_SHIELD_HEIGHT_M: float = 0.20
MIN_FLOOR_GAP_M: float = 0.0
MIN_CEIL_GAP_M: float = 0.0
MIN_REFLECTOR_GAP_M: float = 0.0
MIN_REFLECTOR_THICKNESS_M: float = 0.0
MIN_CHIMNEY_GAP_M: float = 0.0
MAX_CHIMNEY_GAP_M: float = 0.100
MIN_BLOCKAGE_FRACTION: float = 0.0
MAX_BLOCKAGE_FRACTION: float = 1.0
MIN_KILN_STANDOFF_M: float = 0.10
MIN_HOOD_CLEARANCE_M: float = 0.10
MIN_EMISSIVITY: float = 0.01
MAX_EMISSIVITY: float = 0.99


@dataclasses.dataclass(frozen=True)
class SimParams:
  """Immutable SI-unit parameters and derived quantities for a simulation run.

  All lengths are in metres (m), areas in square metres (m^2), temperatures in
  Kelvin (K), angles in radians (rad), flow rates in m^3/s, thermal
  conductivities in W/(m*K), and thermal transmittances in W/(m^2*K).
  """

  experiment_name: str

  # Shield & Wall Geometry (m)
  w_shield_left: float
  w_shield_back: float
  g_floor: float
  g_ceil: float
  t_shield: float
  d_chimney: float
  chimney_blockage_fraction: float
  seal_chimney_sides: bool
  s_kiln_left: float
  s_kiln_back: float
  h_wall: float
  t_gypsum: float
  t_aac: float

  # Kiln, Bung & SSR Geometry (m, rad)
  d_kiln: float
  r_kiln: float
  h_drum: float
  t_lid: float
  h_kiln_base: float
  bung_angle_rad: float
  r_bung: float
  z_bung_offset: float
  bung_captured: bool
  ssr_angle_rad: float
  z_ssr_offset: float
  w_ssr: float
  d_ssr: float
  h_ssr: float

  # Floor, Reflector & Subfloor Stack (m)
  d_reflector: float
  r_reflector: float
  t_reflector_calsil: float
  g_floor_reflector: float
  w_floor_pad: float
  t_floor_pad: float
  t_screed: float
  t_vip: float

  # Hood & Room Enclosure (m, m^3/s)
  d_hood_inlet: float
  r_hood_inlet: float
  d_hood_duct: float
  z_hood_drop: float
  x_hood: float
  y_hood: float
  l_room: float
  w_room: float
  w_glass_door: float
  w_door_open: float
  h_door: float

  # Thermal Sources & Flows (K, W, W/(m^2*K), m^3/s)
  t_internal_k: float
  t_ext_k: float
  u_kiln_side: float
  u_kiln_top: float
  u_kiln_bot: float
  q_bung_open_w: float
  q_ssr_internal_w: float
  v_dot_hood_m3_s: float

  # Surface Emissivities (dimensionless)
  eps_kiln: float
  eps_kiln_side: float
  eps_kiln_top: float
  eps_kiln_bot: float
  eps_shield: float
  eps_gypsum: float
  eps_aac: float
  eps_reflector_top: float
  rho_reflector_top: float
  eps_ssr: float

  # Material Conductivities (W/(m*K)) & Envelope U-Values (W/(m^2*K))
  k_shield: float
  k_gypsum: float
  k_aac: float
  k_pad: float
  k_screed: float
  k_vip: float
  u_roof: float
  u_glass_door: float

  # Discretization Settings (m, count)
  dx_wall: float
  dz_wall: float
  n_theta_kiln: int
  n_z_kiln: int

  # Derived Geometric Coordinates & Elevations (m)
  x_shield_front_back_wall: float
  y_shield_front_left_wall: float
  x_shield_left_start: float
  x_shield_left_end: float
  y_shield_back_start: float
  y_shield_back_end: float
  x_kiln_center: float
  y_kiln_center: float
  h_shield: float
  z_pad_top: float
  z_reflector_bot: float
  z_reflector_top: float
  z_drum_bot: float
  h_kiln_total: float
  z_kiln_top: float
  z_ssr_bot: float
  z_ssr_top: float
  z_bung: float
  x_bung: float
  y_bung: float
  z_mirror: float
  z_hood_inlet: float
  d_chimney_eff: float
  u_aac_wall: float

  def compute_ua_upper_zone(self, z_layer: float) -> float:
    """Computes parallel envelope thermal conductance (UA)_upper in W/K.

    Sums the parallel conductive paths out of the upper hot gas layer through
    the roof/ceiling, the upper portion of the AAC walls, and the upper portion
    of the sliding glass door above height z_layer.

    Args:
      z_layer: Height of the two-zone stratification interface above z = 0 (m).

    Returns:
      Total upper-zone envelope thermal conductance (UA)_upper in W/K.
    """
    z_clamped = max(0.0, min(self.h_wall, z_layer))
    h_upper_wall = self.h_wall - z_clamped
    a_roof = self.l_room * self.w_room
    p_opaque_walls = (
        2.0 * self.l_room
        + 2.0 * self.w_room
        - self.w_glass_door
    )
    a_wall_upper = max(0.0, p_opaque_walls * h_upper_wall)
    h_glass_upper = max(0.0, self.h_door - z_clamped)
    a_glass_upper = self.w_glass_door * h_glass_upper

    return (
        self.u_roof * a_roof
        + self.u_aac_wall * a_wall_upper
        + self.u_glass_door * a_glass_upper
    )

  def compute_ua_lower_zone(self, z_layer: float) -> float:
    """Computes parallel envelope thermal conductance (UA)_lower in W/K.

    Sums the parallel conductive paths out of the lower room zone through the
    lower portion of the AAC walls, the lower portion of the sliding glass door
    below height z_layer, and the exposed room floor outside the kiln footprint.

    Args:
      z_layer: Height of the two-zone stratification interface above z = 0 (m).

    Returns:
      Total lower-zone envelope thermal conductance (UA)_lower in W/K.
    """
    z_clamped = max(0.0, min(self.h_wall, z_layer))
    p_opaque_walls = (
        2.0 * self.l_room
        + 2.0 * self.w_room
        - self.w_glass_door
    )
    a_wall_lower = max(0.0, p_opaque_walls * z_clamped)
    h_glass_lower = max(0.0, min(self.h_door, z_clamped))
    a_glass_lower = self.w_glass_door * h_glass_lower
    a_floor = max(
        0.0,
        self.l_room * self.w_room - math.pi * (self.r_kiln**2),
    )
    r_subfloor = (
        self.t_screed / max(self.k_screed, 1e-4)
        + self.t_vip / max(self.k_vip, 1e-4)
    )
    u_subfloor = 1.0 / max(r_subfloor, 0.1)

    return (
        self.u_aac_wall * a_wall_lower
        + self.u_glass_door * a_glass_lower
        + u_subfloor * a_floor
    )

  def to_derived_proto(
      self,
  ) -> kiln_sim_pb2.SimulationOutput.DerivedParameters:
    """Populates the SimulationOutput.DerivedParameters protobuf message.

    Returns:
      A populated SimulationOutput.DerivedParameters instance with lengths in
      millimetres and thermal transmittance in W/(m^2*K).
    """
    return kiln_sim_pb2.SimulationOutput.DerivedParameters(
        h_shield_mm=self.h_shield * M_TO_MM,
        z_pad_top_mm=self.z_pad_top * M_TO_MM,
        z_reflector_bot_mm=self.z_reflector_bot * M_TO_MM,
        z_reflector_top_mm=self.z_reflector_top * M_TO_MM,
        z_drum_bot_mm=self.z_drum_bot * M_TO_MM,
        h_kiln_total_mm=self.h_kiln_total * M_TO_MM,
        z_kiln_top_mm=self.z_kiln_top * M_TO_MM,
        z_ssr_bot_mm=self.z_ssr_bot * M_TO_MM,
        z_ssr_top_mm=self.z_ssr_top * M_TO_MM,
        z_bung_mm=self.z_bung * M_TO_MM,
        z_mirror_mm=self.z_mirror * M_TO_MM,
        z_hood_inlet_mm=self.z_hood_inlet * M_TO_MM,
        d_chimney_eff_mm=self.d_chimney_eff * M_TO_MM,
        u_aac_wall=self.u_aac_wall,
    )


def validate_params(params: SimParams) -> None:
  """Validates that physical and geometric parameters fall within model bounds.

  Args:
    params: Constructed SimParams instance to validate.

  Raises:
    ValueError: If any parameter violates physical or geometric constraints.
  """
  if params.h_shield <= MIN_SHIELD_HEIGHT_M:
    raise ValueError(
        f"Derived shield height h_shield ({params.h_shield:.3f} m) must be "
        f"greater than {MIN_SHIELD_HEIGHT_M} m. Check h_wall, g_floor, and g_ceil."
    )
  if params.g_floor < MIN_FLOOR_GAP_M - 1e-9:
    raise ValueError(
        f"Floor inlet gap g_floor ({params.g_floor * M_TO_MM:.1f} mm) must be "
        f">= {MIN_FLOOR_GAP_M * M_TO_MM:.1f} mm."
    )
  if params.g_ceil < MIN_CEIL_GAP_M - 1e-9:
    raise ValueError(
        f"Ceiling outlet gap g_ceil ({params.g_ceil * M_TO_MM:.1f} mm) must be "
        f">= {MIN_CEIL_GAP_M * M_TO_MM:.1f} mm."
    )
  if (
      params.s_kiln_left < MIN_KILN_STANDOFF_M
      or params.s_kiln_back < MIN_KILN_STANDOFF_M
  ):
    raise ValueError(
        "Kiln standoff s_kiln_left and s_kiln_back must be >= "
        f"{MIN_KILN_STANDOFF_M * M_TO_MM:.1f} mm so opposing turbulent "
        "boundary layers do not merge."
    )
  if (
      params.d_chimney < MIN_CHIMNEY_GAP_M - 1e-9
      or params.d_chimney > MAX_CHIMNEY_GAP_M + 1e-9
  ):
    raise ValueError(
        f"Chimney gap d_chimney ({params.d_chimney * M_TO_MM:.1f} mm) is "
        f"outside the valid [{MIN_CHIMNEY_GAP_M * M_TO_MM:.1f} mm, "
        f"{MAX_CHIMNEY_GAP_M * M_TO_MM:.1f} mm] range."
    )
  if not (
      MIN_BLOCKAGE_FRACTION
      <= params.chimney_blockage_fraction
      <= MAX_BLOCKAGE_FRACTION
  ):
    raise ValueError(
        "chimney_blockage_fraction must lie within "
        f"[{MIN_BLOCKAGE_FRACTION}, {MAX_BLOCKAGE_FRACTION}]."
    )
  if params.d_reflector < -1e-9:
    raise ValueError(
        f"Reflector diameter d_reflector ({params.d_reflector * M_TO_MM:.1f} mm) "
        "must be >= 0.0 mm."
    )
  if params.g_floor_reflector < MIN_REFLECTOR_GAP_M - 1e-9:
    raise ValueError(
        f"Reflector floor gap g_floor_reflector "
        f"({params.g_floor_reflector * M_TO_MM:.1f} mm) must be >= "
        f"{MIN_REFLECTOR_GAP_M * M_TO_MM:.1f} mm."
    )
  if params.t_reflector_calsil < MIN_REFLECTOR_THICKNESS_M - 1e-9:
    raise ValueError(
        f"Reflector thickness t_reflector_calsil "
        f"({params.t_reflector_calsil * M_TO_MM:.1f} mm) must be >= "
        f"{MIN_REFLECTOR_THICKNESS_M * M_TO_MM:.1f} mm."
    )
  if params.r_reflector > 0.0 and params.z_reflector_top >= params.z_drum_bot - 1e-9:
    raise ValueError(
        f"Reflector top elevation ({params.z_reflector_top * M_TO_MM:.1f} mm) "
        f"must be strictly below the kiln bottom plate "
        f"({params.z_drum_bot * M_TO_MM:.1f} mm)."
    )
  min_pad_x = params.x_kiln_center + params.r_kiln
  min_pad_y = params.y_kiln_center + params.r_kiln
  if (
      params.w_floor_pad < min_pad_x - 1e-9
      or params.w_floor_pad < min_pad_y - 1e-9
  ):
    raise ValueError(
        f"Square floor pad width w_floor_pad "
        f"({params.w_floor_pad * M_TO_MM:.1f} mm) must extend fully under all "
        f"sides of the kiln (requires >= "
        f"{max(min_pad_x, min_pad_y) * M_TO_MM:.1f} mm)."
    )
  if params.w_floor_pad < params.d_reflector:
    logging.warning(
        "Square floor pad width (%.1f mm) is smaller than reflector diameter "
        "(%.1f mm); reflector overhangs vinyl floor.",
        params.w_floor_pad * M_TO_MM,
        params.d_reflector * M_TO_MM,
    )
  if params.z_hood_inlet <= params.z_kiln_top + MIN_HOOD_CLEARANCE_M:
    raise ValueError(
        f"Hood inlet elevation ({params.z_hood_inlet:.3f} m) must sit at least "
        f"{MIN_HOOD_CLEARANCE_M} m above the closed kiln lid "
        f"({params.z_kiln_top:.3f} m)."
    )
  for name, eps in [
      ("eps_kiln_side", params.eps_kiln_side),
      ("eps_kiln_top", params.eps_kiln_top),
      ("eps_kiln_bot", params.eps_kiln_bot),
      ("eps_shield", params.eps_shield),
      ("eps_gypsum", params.eps_gypsum),
      ("eps_aac", params.eps_aac),
      ("eps_reflector_top", params.eps_reflector_top),
      ("eps_ssr", params.eps_ssr),
  ]:
    if not (MIN_EMISSIVITY <= eps <= MAX_EMISSIVITY):
      raise ValueError(
          f"Emissivity {name}={eps} must lie in "
          f"[{MIN_EMISSIVITY}, {MAX_EMISSIVITY}]."
      )


def from_proto(config: kiln_sim_pb2.SimulationConfig) -> SimParams:
  """Constructs and validates a SimParams instance from a SimulationConfig proto.

  Args:
    config: Parsed SimulationConfig protobuf message.

  Returns:
    A validated, immutable SimParams instance in SI units.

  Raises:
    ValueError: If any parameter is invalid or out of physical bounds.
  """
  geom = config.geometry
  shield = geom.shield
  room = geom.room
  chimney = room.chimney
  kiln = geom.kiln
  bung = geom.bung
  ssr = geom.ssr
  floor = geom.floor
  hood = geom.hood

  therm = config.thermal
  sources = therm.sources
  emis = therm.emissivities
  cond = therm.conductivities
  grid = config.grid

  # Convert primary geometry to metres.
  w_shield_left = shield.w_shield_left_mm * MM_TO_M
  w_shield_back = shield.w_shield_back_mm * MM_TO_M
  g_floor = shield.g_floor_mm * MM_TO_M
  g_ceil = shield.g_ceil_mm * MM_TO_M
  t_shield = shield.t_shield_mm * MM_TO_M

  d_chimney = chimney.d_chimney_mm * MM_TO_M
  blockage = chimney.chimney_blockage_fraction
  seal_sides = chimney.seal_chimney_sides

  h_wall = room.h_wall_mm * MM_TO_M
  t_gypsum = room.t_gypsum_mm * MM_TO_M
  t_aac = room.t_aac_mm * MM_TO_M
  l_room = room.l_room_mm * MM_TO_M
  w_room = room.w_room_mm * MM_TO_M
  w_glass_door = room.w_glass_door_mm * MM_TO_M
  w_door_open = room.w_door_open_mm * MM_TO_M
  h_door = room.h_door_mm * MM_TO_M

  s_kiln_left = kiln.s_kiln_left_mm * MM_TO_M
  s_kiln_back = kiln.s_kiln_back_mm * MM_TO_M
  d_kiln = kiln.d_kiln_mm * MM_TO_M
  r_kiln = 0.5 * d_kiln
  h_drum = kiln.h_drum_mm * MM_TO_M
  t_lid = kiln.t_lid_mm * MM_TO_M
  h_kiln_base = kiln.h_kiln_base_mm * MM_TO_M

  bung_angle_rad = math.radians(bung.bung_angle_deg)
  r_bung = bung.r_bung_mm * MM_TO_M
  z_bung_offset = bung.z_bung_offset_mm * MM_TO_M

  ssr_angle_rad = math.radians(ssr.ssr_angle_deg)
  z_ssr_offset = ssr.z_ssr_offset_mm * MM_TO_M
  w_ssr = ssr.w_ssr_mm * MM_TO_M
  d_ssr = ssr.d_ssr_mm * MM_TO_M
  h_ssr = ssr.h_ssr_mm * MM_TO_M

  d_reflector = floor.d_reflector_mm * MM_TO_M
  r_reflector = 0.5 * d_reflector
  t_reflector_calsil = floor.t_reflector_calsil_mm * MM_TO_M
  g_floor_reflector = floor.g_floor_reflector_mm * MM_TO_M
  w_floor_pad = floor.w_floor_pad_mm * MM_TO_M
  t_floor_pad = floor.t_floor_pad_mm * MM_TO_M
  t_screed = floor.t_screed_mm * MM_TO_M
  t_vip = floor.t_vip_mm * MM_TO_M

  d_hood_inlet = hood.d_hood_inlet_mm * MM_TO_M
  r_hood_inlet = 0.5 * d_hood_inlet
  d_hood_duct = hood.d_hood_duct_mm * MM_TO_M
  z_hood_drop = hood.z_hood_drop_mm * MM_TO_M
  x_hood = hood.x_hood_mm * MM_TO_M
  y_hood = hood.y_hood_mm * MM_TO_M

  # Derived horizontal coordinates from bare AAC wall datum planes (x=0, y=0).
  x_shield_front_back_wall = d_chimney + t_shield
  y_shield_front_left_wall = d_chimney + t_shield
  x_shield_left_start = d_chimney + t_shield
  x_shield_left_end = x_shield_left_start + w_shield_left
  y_shield_back_start = d_chimney
  y_shield_back_end = y_shield_back_start + w_shield_back

  x_kiln_center = x_shield_front_back_wall + s_kiln_back + r_kiln
  y_kiln_center = y_shield_front_left_wall + s_kiln_left + r_kiln

  # Derived vertical datum stack starting from z = 0 at top of structural screed.
  # Both the kiln stand legs and the floor reflector assembly rest independently
  # on top of the floor pad (z_pad_top).
  h_shield = h_wall - g_floor - g_ceil
  z_pad_top = t_floor_pad
  z_reflector_bot = z_pad_top + g_floor_reflector
  z_reflector_top = z_reflector_bot + t_reflector_calsil
  z_drum_bot = z_pad_top + h_kiln_base
  h_kiln_total = h_drum + t_lid
  z_kiln_top = z_drum_bot + h_kiln_total
  z_ssr_bot = z_drum_bot + z_ssr_offset
  z_ssr_top = z_ssr_bot + h_ssr
  z_bung = z_drum_bot + z_bung_offset
  x_bung = x_kiln_center + r_bung * math.cos(bung_angle_rad)
  y_bung = y_kiln_center + r_bung * math.sin(bung_angle_rad)
  z_mirror = 2.0 * z_reflector_top - z_drum_bot
  z_hood_inlet = h_wall - z_hood_drop

  # Derived hydraulic and conductive parameters.
  d_chimney_eff = d_chimney * (1.0 - blockage)
  u_aac_wall = 1.0 / (R_SI_WALL + (t_aac / cond.k_aac) + R_SE_WALL)

  eps_kiln_side = (
      emis.eps_kiln_side if emis.eps_kiln_side > 0.0 else emis.eps_kiln
  )
  eps_kiln_top = (
      emis.eps_kiln_top if emis.eps_kiln_top > 0.0 else emis.eps_kiln
  )
  eps_kiln_bot = (
      emis.eps_kiln_bot if emis.eps_kiln_bot > 0.0 else emis.eps_kiln
  )

  params = SimParams(
      experiment_name=config.experiment_name or "unnamed_experiment",
      w_shield_left=w_shield_left,
      w_shield_back=w_shield_back,
      g_floor=g_floor,
      g_ceil=g_ceil,
      t_shield=t_shield,
      d_chimney=d_chimney,
      chimney_blockage_fraction=blockage,
      seal_chimney_sides=seal_sides,
      s_kiln_left=s_kiln_left,
      s_kiln_back=s_kiln_back,
      h_wall=h_wall,
      t_gypsum=t_gypsum,
      t_aac=t_aac,
      d_kiln=d_kiln,
      r_kiln=r_kiln,
      h_drum=h_drum,
      t_lid=t_lid,
      h_kiln_base=h_kiln_base,
      bung_angle_rad=bung_angle_rad,
      r_bung=r_bung,
      z_bung_offset=z_bung_offset,
      bung_captured=bung.bung_captured,
      ssr_angle_rad=ssr_angle_rad,
      z_ssr_offset=z_ssr_offset,
      w_ssr=w_ssr,
      d_ssr=d_ssr,
      h_ssr=h_ssr,
      d_reflector=d_reflector,
      r_reflector=r_reflector,
      t_reflector_calsil=t_reflector_calsil,
      g_floor_reflector=g_floor_reflector,
      w_floor_pad=w_floor_pad,
      t_floor_pad=t_floor_pad,
      t_screed=t_screed,
      t_vip=t_vip,
      d_hood_inlet=d_hood_inlet,
      r_hood_inlet=r_hood_inlet,
      d_hood_duct=d_hood_duct,
      z_hood_drop=z_hood_drop,
      x_hood=x_hood,
      y_hood=y_hood,
      l_room=l_room,
      w_room=w_room,
      w_glass_door=w_glass_door,
      w_door_open=w_door_open,
      h_door=h_door,
      t_internal_k=sources.t_internal_c + KELVIN_OFFSET,
      t_ext_k=sources.t_ext_c + KELVIN_OFFSET,
      u_kiln_side=sources.u_kiln_side,
      u_kiln_top=sources.u_kiln_top,
      u_kiln_bot=sources.u_kiln_bot,
      q_bung_open_w=sources.q_bung_open_w,
      q_ssr_internal_w=sources.q_ssr_internal_w,
      v_dot_hood_m3_s=sources.exhaust_fan_cfm * CFM_TO_M3_S,
      eps_kiln=eps_kiln_side,
      eps_kiln_side=eps_kiln_side,
      eps_kiln_top=eps_kiln_top,
      eps_kiln_bot=eps_kiln_bot,
      eps_shield=emis.eps_shield,
      eps_gypsum=emis.eps_gypsum,
      eps_aac=emis.eps_aac,
      eps_reflector_top=emis.eps_reflector_top,
      rho_reflector_top=1.0 - emis.eps_reflector_top,
      eps_ssr=emis.eps_ssr,
      k_shield=cond.k_shield,
      k_gypsum=cond.k_gypsum,
      k_aac=cond.k_aac,
      k_pad=cond.k_pad,
      k_screed=cond.k_screed,
      k_vip=cond.k_vip,
      u_roof=cond.u_roof,
      u_glass_door=cond.u_glass_door,
      dx_wall=grid.dx_wall_mm * MM_TO_M,
      dz_wall=grid.dz_wall_mm * MM_TO_M,
      n_theta_kiln=int(grid.n_theta_kiln),
      n_z_kiln=int(grid.n_z_kiln),
      x_shield_front_back_wall=x_shield_front_back_wall,
      y_shield_front_left_wall=y_shield_front_left_wall,
      x_shield_left_start=x_shield_left_start,
      x_shield_left_end=x_shield_left_end,
      y_shield_back_start=y_shield_back_start,
      y_shield_back_end=y_shield_back_end,
      x_kiln_center=x_kiln_center,
      y_kiln_center=y_kiln_center,
      h_shield=h_shield,
      z_pad_top=z_pad_top,
      z_reflector_bot=z_reflector_bot,
      z_reflector_top=z_reflector_top,
      z_drum_bot=z_drum_bot,
      h_kiln_total=h_kiln_total,
      z_kiln_top=z_kiln_top,
      z_ssr_bot=z_ssr_bot,
      z_ssr_top=z_ssr_top,
      z_bung=z_bung,
      x_bung=x_bung,
      y_bung=y_bung,
      z_mirror=z_mirror,
      z_hood_inlet=z_hood_inlet,
      d_chimney_eff=d_chimney_eff,
      u_aac_wall=u_aac_wall,
  )

  validate_params(params)
  return params