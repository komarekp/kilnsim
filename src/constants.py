"""Shared physical, geometric, numerical, and engineering safety constants.

Single source of truth imported across physics modules, mesh discretization,
post-processing metrics, and parametric sweep configuration builders so
constants are never duplicated across files.
"""

# ---------------------------------------------------------------------------
# Fundamental physical constants & unit conversions (SI units)
# ---------------------------------------------------------------------------
KELVIN_OFFSET: float = 273.15
MM_TO_M: float = 1.0e-3
M_TO_MM: float = 1.0e3
CFM_TO_M3_S: float = 4.7194745e-4
STEFAN_BOLTZMANN: float = 5.670374419e-8  # W / (m^2 * K^4)
GRAVITY: float = 9.80665  # m / s^2

# ---------------------------------------------------------------------------
# Dry air thermophysical properties at ~315 K and 1 atm
# ---------------------------------------------------------------------------
AIR_RHO_REF: float = 1.12  # kg / m^3
AIR_CP: float = 1007.0  # J / (kg * K)
AIR_MU: float = 1.91e-5  # Pa * s
AIR_NU: float = 1.70e-5  # m^2 / s
AIR_K: float = 0.0274  # W / (m * K)
AIR_PR: float = 0.705  # Prandtl number
AIR_NU_REF_300K: float = 1.57e-5  # m^2 / s kinematic viscosity at 300 K
AIR_K_REF_300K: float = 0.0262  # W / (m * K) thermal conductivity at 300 K
AIR_T_REF_K: float = 300.0  # Reference temperature for air scaling (K)

# ---------------------------------------------------------------------------
# Standard film resistances & internal / interface convection coefficients
# ---------------------------------------------------------------------------
R_SI_WALL: float = 0.13  # m^2 * K / W interior wall film resistance
R_SE_WALL: float = 0.04  # m^2 * K / W exterior wall film resistance
H_CONV_MIN_W_M2K: float = 2.0  # W / (m^2 * K) natural convection floor clamp
H_CONV_UNDER_REFLECTOR_W_M2K: float = 3.0  # W / (m^2 * K) under elevated reflector
H_CONV_SSR_INTERNAL_W_M2K: float = 5.5  # W / (m^2 * K) internal SSR housing wall
H_CONV_CEILING_JET_MIN_W_M2K: float = 2.0  # W / (m^2 * K) downward-facing floor
H_CONV_HORIZ_UNSTABLE_COEFF: float = 1.52
H_CONV_HORIZ_STABLE_COEFF: float = 0.65
H_CONV_STABLE_MIN_W_M2K: float = 0.5
H_CONV_VERT_SENSOR_COEFF: float = 1.42

# SSR louvered enclosure model constants
SSR_LOUVER_EFF_AREA_M2: float = 0.0022  # m^2 effective free louver area
SSR_LOCAL_INFLOW_KG_S: float = 0.035  # kg / s local convective entrainment
SSR_LOUVER_DISCHARGE_COEFF: float = 0.55
SSR_LOUVER_MIN_FLOW_KG_S: float = 1.2e-3  # kg / s minimum background leakage
SSR_BAFFLE_CONV_W_M2K: float = 6.5  # W / (m^2 * K) ventilated standoff gap
SSR_FACING_COS_MIN: float = 0.70710678  # cos(45 deg) kiln patch selection
SSR_SUBKILN_CONV_WEIGHT: float = 0.50  # sub-kiln bottom air entering SSR louver

# ---------------------------------------------------------------------------
# Validation bounds in SI units (metres and dimensionless)
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Mesh discretization & sensor placement constants (SI units)
# ---------------------------------------------------------------------------
EXPOSED_GYPSUM_BAND_WIDTH_M: float = 1.0
NUM_SINGLE_CAP_PATCHES: int = 3  # KILN_LID + KILN_BOTTOM + REFLECTOR_TOP
NUM_SSR_FACES: int = 3  # Rear + 2 side exposed faces on SSR_BOX
FLOOR_SENSOR_EDGE_OFFSET_M: float = 0.005
CEILING_SENSOR_RIM_OFFSET_M: float = 0.010
CEILING_SENSOR_WALL_OFFSET_M: float = 0.050
CEILING_NUM_AZIMUTHS: int = 8
CEILING_GRID_MAX_RADIUS_M: float = 1.80
CEILING_PLUME_INNER_DECAY_M: float = 0.15
CEILING_CHIMNEY_JET_DECAY_M: float = 0.25
GEOM_TOL_M: float = 1.0e-4

# ---------------------------------------------------------------------------
# Radiative view factor quadrature & clipping constants
# ---------------------------------------------------------------------------
KILN_DISK_QUAD_R: int = 16
KILN_DISK_QUAD_THETA: int = 48
SSR_QUAD_U: int = 6
SSR_QUAD_V: int = 6
MAX_PATCH_VF_ROW_SUM: float = 1.0
MAX_SENSOR_VF_SUM: float = 0.99

# ---------------------------------------------------------------------------
# Plume, chimney, hood & two-zone room constants
# ---------------------------------------------------------------------------
CHIMNEY_LAMINAR_NU_DH: float = 7.54
CHIMNEY_CAVITY_CONV_COEFF: float = 0.42
CHIMNEY_CAVITY_PR_EXP: float = 0.012
CHIMNEY_CAVITY_ASPECT_EXP: float = -0.3
CHIMNEY_CEILING_MIXING_FRACTION: float = 0.35
PLUME_CEILING_MIXING_FRACTION: float = 0.35
KILN_BOT_PLUME_FRACTION: float = 0.65
MTT_PLUME_FLOW_COEFF: float = 0.071
MTT_SOURCE_ENTRAINMENT_COEFF: float = 0.018
HESKESTAD_VIRTUAL_ORIGIN_DIAM_COEFF: float = 1.02
HESKESTAD_VIRTUAL_ORIGIN_HEAT_COEFF: float = 0.083
GAUSSIAN_PLUME_SPREAD_COEFF: float = 0.12
GAUSSIAN_95_RADIUS_MULTIPLIER: float = 2.45
DOOR_ORIFICE_DISCHARGE_COEFF: float = 0.65
ROOM_INFILTRATION_AREA_M2: float = 0.005
FAN_MAX_STATIC_PRESSURE_PA: float = 250.0
MIN_DUCT_MASS_FLOW_KG_S: float = 1.0e-6
HOOD_ENTRY_LOSS_COEFF: float = 0.25
DUCT_FRICTION_FACTOR: float = 0.020
DUCT_EQUIV_LENGTH_M: float = 3.0
DUCT_EXIT_LOSS_COEFF: float = 1.0
HOOD_REF_INLET_RADIUS_M: float = 0.40
DOOR_FLOODING_FROUDE_COEFF: float = 0.55
HOOD_SUCTION_REACH_COEFF: float = 0.22
HOOD_SUCTION_MAX_FRAC: float = 0.60

# ---------------------------------------------------------------------------
# Engineering pass / fail limits (deg C) and floor emissivities
# ---------------------------------------------------------------------------
LIMIT_GYPSUM_C: float = 40.0
LIMIT_CEILING_C: float = 40.0
LIMIT_SSR_AIR_C: float = 35.0
LIMIT_SSR_CASE_C: float = 45.0
LIMIT_VIP_C: float = 50.0
LIMIT_VINYL_C: float = 45.0
LIMIT_SHIELD_C: float = 250.0
LIMIT_AAC_C: float = 120.0

FLOOR_PAD_EMISSIVITY: float = 0.88
VINYL_FLOOR_EMISSIVITY: float = 0.90

# ---------------------------------------------------------------------------
# Coupled steady-state solver numerical parameters
# ---------------------------------------------------------------------------
SOLVER_MAX_OUTER_ITERS: int = 25
SOLVER_MAX_INNER_ITERS: int = 60
SOLVER_INNER_RES_TOL_W_M2: float = 1.0e-4
SOLVER_OUTER_RES_TOL_W_M2: float = 0.5
SOLVER_OUTER_TEMP_TOL_K: float = 0.005
SOLVER_OUTER_FAST_TEMP_TOL_K: float = 0.001
SOLVER_OUTER_LAYER_TOL_M: float = 0.002
SOLVER_MIN_TEMP_K: float = 250.0
SOLVER_MAX_STEP_K: float = 25.0

# ---------------------------------------------------------------------------
# Reference Rohde Ecotop 60 studio & sweep default configuration values
# ---------------------------------------------------------------------------
DEFAULT_SHIELD_THICKNESS_MM: float = 12.5
DEFAULT_WALL_HEIGHT_MM: float = 2345.0
DEFAULT_GYPSUM_THICKNESS_MM: float = 12.5
DEFAULT_AAC_THICKNESS_MM: float = 215.0
DEFAULT_ROOM_LENGTH_MM: float = 3530.0
DEFAULT_ROOM_WIDTH_MM: float = 2770.0
DEFAULT_GLASS_DOOR_WIDTH_MM: float = 2000.0
DEFAULT_DOOR_HEIGHT_MM: float = 2100.0
DEFAULT_CHIMNEY_BLOCKAGE_FRACTION: float = 0.08

DEFAULT_KILN_DIAMETER_MM: float = 700.0
DEFAULT_KILN_DRUM_HEIGHT_MM: float = 595.0
DEFAULT_KILN_LID_THICKNESS_MM: float = 75.0
DEFAULT_KILN_BASE_HEIGHT_MM: float = 210.0

DEFAULT_BUNG_ANGLE_DEG: float = 0.0
DEFAULT_BUNG_RADIUS_MM: float = 415.0
DEFAULT_BUNG_Z_OFFSET_MM: float = 540.0

DEFAULT_SSR_ANGLE_DEG: float = -45.0
DEFAULT_SSR_Z_OFFSET_MM: float = 100.0
DEFAULT_SSR_WIDTH_MM: float = 190.0
DEFAULT_SSR_DEPTH_MM: float = 146.5
DEFAULT_SSR_HEIGHT_MM: float = 430.0

DEFAULT_SCREED_THICKNESS_MM: float = 40.0
DEFAULT_VIP_THICKNESS_MM: float = 20.0
DEFAULT_HOOD_DUCT_DIAMETER_MM: float = 200.0

DEFAULT_KILN_INTERNAL_C: float = 1280.0
DEFAULT_EXT_TEMP_C: float = 20.0
DEFAULT_U_KILN_TOP: float = 2.00
DEFAULT_Q_BUNG_OPEN_W: float = 110.0
DEFAULT_Q_SSR_INTERNAL_W: float = 18.8

DEFAULT_EPS_KILN_SIDE: float = 0.60
DEFAULT_EPS_KILN_TOP: float = 0.85
DEFAULT_EPS_KILN_BOT: float = 0.15
DEFAULT_EPS_SHIELD: float = 0.90
DEFAULT_EPS_GYPSUM: float = 0.90
DEFAULT_EPS_AAC: float = 0.88
DEFAULT_EPS_REFLECTOR_TOP: float = 0.15
DEFAULT_EPS_SSR: float = 0.60

DEFAULT_K_SHIELD: float = 0.175
DEFAULT_K_GYPSUM: float = 0.24
DEFAULT_K_AAC: float = 0.13
DEFAULT_K_PAD: float = 1.40
DEFAULT_K_SCREED: float = 1.20
DEFAULT_K_VIP: float = 0.007
DEFAULT_U_ROOF: float = 0.18
DEFAULT_U_GLASS_DOOR: float = 1.40

DEFAULT_DX_WALL_MM: float = 50.0
DEFAULT_DZ_WALL_MM: float = 50.0
DEFAULT_N_THETA_KILN: int = 24
DEFAULT_N_Z_KILN: int = 6

