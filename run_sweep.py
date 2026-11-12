#!/usr/bin/env python3
"""Parametric sweep generator and runner for the kiln thermal simulator.

Generates SimulationConfig .textproto files in sweep_configs/ and writes the
corresponding SimulationOutput .out.textproto files in sweep_results/.
"""

from concurrent import futures
import csv
import dataclasses
import datetime
import itertools
import os
import pathlib
import random
import re
import sys
import time
from typing import Iterable, Optional, Sequence

_SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
_WORKSPACE_DIR = pathlib.Path(
    os.environ.get("BUILD_WORKSPACE_DIRECTORY", str(_SCRIPT_DIR))
).resolve()

try:
  # Verify we are running inside the Bazel hermetic Python environment with
  # generated protobuf modules available. If invoked directly via host python3,
  # transparently re-exec through `bazel run //:run_sweep`.
  from google.protobuf import runtime_version as _  # pylint: disable=g-import-not-at-top,unused-import
  from google.protobuf import text_format  # pylint: disable=g-import-not-at-top
  from proto import kiln_sim_pb2  # pylint: disable=g-import-not-at-top
except ImportError:
  os.chdir(_WORKSPACE_DIR)
  os.execvp("bazel", ["bazel", "run", "//:run_sweep", "--", *sys.argv[1:]])

from absl import app  # pylint: disable=g-import-not-at-top
from absl import flags  # pylint: disable=g-import-not-at-top
from absl import logging  # pylint: disable=g-import-not-at-top
from src import io_manager  # pylint: disable=g-import-not-at-top
from src import mesh as mesh_lib  # pylint: disable=g-import-not-at-top
from src import metrics as metrics_lib  # pylint: disable=g-import-not-at-top
from src import params as params_lib  # pylint: disable=g-import-not-at-top
from src import solver as solver_lib  # pylint: disable=g-import-not-at-top
from src import view_factors as vf_lib  # pylint: disable=g-import-not-at-top

_NUM_CPU = os.cpu_count() or 4

_CONFIG_DIR = flags.DEFINE_string(
    "config_dir",
    str(_WORKSPACE_DIR / "sweep_configs"),
    "Directory where generated .textproto configuration files are written.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    str(_WORKSPACE_DIR / "sweep_results"),
    "Directory where simulation .out.textproto result files are written.",
)
_JOBS = flags.DEFINE_integer(
    "jobs",
    _NUM_CPU // 2,
    "Number of parallel worker processes for executing the sweep.",
)
_CONFIGS_ONLY = flags.DEFINE_boolean(
    "configs_only",
    False,
    "If True, only generate the .textproto files in config_dir without running "
    "the solver.",
)
_QUICK = flags.DEFINE_boolean(
    "quick",
    False,
    "If True, sample 2 * max_workers simulations using reservoir sampling for "
    "a quick end-to-end smoketest.",
)
_CSV_OUTPUT = flags.DEFINE_string(
    "csv_output",
    str(_WORKSPACE_DIR / "sweep_results.csv"),
    "Path where the summary CSV of sweep parameters and metrics is written.",
)
_CSV_ONLY = flags.DEFINE_boolean(
    "csv_only",
    False,
    "If True, only aggregate existing .out.textproto files in output_dir into "
    "csv_output without running simulations.",
)
_DRY_RUN = flags.DEFINE_boolean(
    "dry_run",
    False,
    "If True, print only the output filenames of the runs that would be "
    "executed (skipping already-completed well-formed outputs) and exit.",
)

# Swept parameter domains.
KILN_STANDOFFS_MM: tuple[float, ...] = (200.0, 300.0)
U_KILN_SIDE_VALUES: tuple[float, ...] = (0.8, 0.5, 0.3)
PAD_SIZES_MM: tuple[float, ...] = (1200.0,)
PAD_THICKNESSES_MM: tuple[float, ...] = (0.0,)
DOOR_OPEN_VALUES_MM: tuple[float, ...] = (0.0, 150.0, 800.0)
CFM_VALUES: tuple[float, ...] = (350.0, 400.0, 450.0)
REFLECTOR_DIAMETERS_MM: tuple[float, ...] = (700.0, 800.0, 900.0, 1000.0)
REFLECTOR_CALSIL_THICKNESSES_MM: tuple[float, ...] = (0.0, 10.0)
REFLECTOR_GAPS_MM: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
CHIMNEY_DEPTHS_MM: tuple[float, ...] = (0.0,)
CHIMNEY_GAPS_MM: tuple[float, ...] = (0.0,)
SHIELD_WIDTHS_MM: tuple[float, ...] = (1200.0,)
HOOD_INLETS_MM: tuple[float, ...] = (700.0, 800.0)
BUNG_CAPTURED_VALUES: tuple[bool, ...] = (True,)


def _fmt_num(val: float) -> str:
  """Formats a numeric parameter value concisely for filenames."""
  if float(val).is_integer():
    return str(int(round(val)))
  return f"{val:g}"


@dataclasses.dataclass(frozen=True)
class SweepPoint:
  """Single parameter combination in the parametric sweep."""

  s_kiln_mm: float
  w_floor_pad_mm: float
  t_floor_pad_mm: float
  w_door_open_mm: float
  exhaust_fan_cfm: float
  d_reflector_mm: float
  t_reflector_calsil_mm: float
  g_floor_reflector_mm: float
  d_chimney_mm: float
  g_chimney_mm: float
  w_shield_mm: float
  d_hood_inlet_mm: float
  bung_captured: bool
  u_kiln_side: float = 0.8

  @property
  def experiment_name(self) -> str:
    """Returns a deterministic experiment identifier and filename stem."""
    return (
        f"standoff{_fmt_num(self.s_kiln_mm)}_"
        f"ukiln{_fmt_num(self.u_kiln_side)}_"
        f"pad{_fmt_num(self.w_floor_pad_mm)}x{_fmt_num(self.t_floor_pad_mm)}_"
        f"refl{_fmt_num(self.d_reflector_mm)}_"
        f"calsil{_fmt_num(self.t_reflector_calsil_mm)}_"
        f"gap{_fmt_num(self.g_floor_reflector_mm)}_"
        f"hood{_fmt_num(self.d_hood_inlet_mm)}_"
        f"door{_fmt_num(self.w_door_open_mm)}_"
        f"cfm{_fmt_num(self.exhaust_fan_cfm)}"
    )

  @property
  def geometry_cache_key(self) -> tuple[float, ...]:
    """Returns the subset of parameters that determine mesh and view factors."""
    return (
        self.s_kiln_mm,
        self.w_floor_pad_mm,
        self.t_floor_pad_mm,
        self.d_reflector_mm,
        self.t_reflector_calsil_mm,
        self.g_floor_reflector_mm,
        self.d_chimney_mm,
        self.g_chimney_mm,
        self.w_shield_mm,
        self.d_hood_inlet_mm,
    )


@dataclasses.dataclass(frozen=True)
class FailedRun:
  """Details of a single sweep configuration that failed or did not converge."""

  experiment_name: str
  config_path: str
  reason: str


def enumerate_sweep_points() -> list[SweepPoint]:
  """Enumerates all valid parameter combinations in the sweep."""
  points: list[SweepPoint] = []
  for (
      s_kiln,
      u_kiln_side,
      w_pad,
      t_pad,
      d_refl,
      g_refl,
      t_calsil,
      d_chim,
      g_chim,
      w_shield,
      d_hood,
      bung_cap,
      w_door,
      cfm,
  ) in itertools.product(
      KILN_STANDOFFS_MM,
      U_KILN_SIDE_VALUES,
      PAD_SIZES_MM,
      PAD_THICKNESSES_MM,
      REFLECTOR_DIAMETERS_MM,
      REFLECTOR_GAPS_MM,
      REFLECTOR_CALSIL_THICKNESSES_MM,
      CHIMNEY_DEPTHS_MM,
      CHIMNEY_GAPS_MM,
      SHIELD_WIDTHS_MM,
      HOOD_INLETS_MM,
      BUNG_CAPTURED_VALUES,
      DOOR_OPEN_VALUES_MM,
      CFM_VALUES,
  ):
    # Constraint: when there is no floor reflector (d_reflector == 0), only the
    # zero-gap, zero-CalSil combination is meaningful.
    if d_refl == 0.0 and (g_refl != 0.0 or t_calsil != 0.0):
      continue
    # Constraint: if reflector gap < 10 mm, reflector CalSil thickness must be 0.
    if g_refl < 10.0 and t_calsil != 0.0:
      continue
    points.append(
        SweepPoint(
            s_kiln_mm=s_kiln,
            u_kiln_side=u_kiln_side,
            w_floor_pad_mm=w_pad,
            t_floor_pad_mm=t_pad,
            w_door_open_mm=w_door,
            exhaust_fan_cfm=cfm,
            d_reflector_mm=d_refl,
            t_reflector_calsil_mm=t_calsil,
            g_floor_reflector_mm=g_refl,
            d_chimney_mm=d_chim,
            g_chimney_mm=g_chim,
            w_shield_mm=w_shield,
            d_hood_inlet_mm=d_hood,
            bung_captured=bung_cap,
        )
    )
  return points


def reservoir_sample_points(
    points: Iterable[SweepPoint],
    sample_size: int,
    seed: Optional[int] = None,
) -> list[SweepPoint]:
  """Samples `sample_size` points uniformly without replacement via Algorithm R.

  Args:
    points: Iterable of SweepPoint instances to sample from.
    sample_size: Target reservoir size (e.g., 2 * max_workers).
    seed: Optional random seed for deterministic sampling in tests.

  Returns:
    List of up to `sample_size` sampled SweepPoint instances.
  """
  if sample_size <= 0:
    return []
  rng = random.Random(seed)
  reservoir: list[SweepPoint] = []
  for idx, point in enumerate(points):
    if idx < sample_size:
      reservoir.append(point)
    else:
      replace_idx = rng.randint(0, idx)
      if replace_idx < sample_size:
        reservoir[replace_idx] = point
  return reservoir


def build_config_proto(point: SweepPoint) -> kiln_sim_pb2.SimulationConfig:
  """Builds a SimulationConfig protobuf for the given sweep point."""
  cfg = kiln_sim_pb2.SimulationConfig()
  cfg.experiment_name = point.experiment_name

  # Shield geometry (12.5 mm thickness from recent 40screed baseline).
  cfg.geometry.shield.w_shield_left_mm = point.w_shield_mm
  cfg.geometry.shield.w_shield_back_mm = point.w_shield_mm
  cfg.geometry.shield.g_floor_mm = point.g_chimney_mm
  cfg.geometry.shield.g_ceil_mm = point.g_chimney_mm
  cfg.geometry.shield.t_shield_mm = 12.5

  # Room and behind-shield chimney geometry.
  cfg.geometry.room.h_wall_mm = 2345.0
  cfg.geometry.room.t_gypsum_mm = 12.5
  cfg.geometry.room.t_aac_mm = 215.0
  cfg.geometry.room.l_room_mm = 3530.0
  cfg.geometry.room.w_room_mm = 2770.0
  cfg.geometry.room.w_glass_door_mm = 2000.0
  cfg.geometry.room.w_door_open_mm = point.w_door_open_mm
  cfg.geometry.room.h_door_mm = 2100.0

  cfg.geometry.room.chimney.d_chimney_mm = point.d_chimney_mm
  cfg.geometry.room.chimney.chimney_blockage_fraction = 0.08
  cfg.geometry.room.chimney.seal_chimney_sides = True

  # Kiln geometry.
  cfg.geometry.kiln.s_kiln_left_mm = point.s_kiln_mm
  cfg.geometry.kiln.s_kiln_back_mm = point.s_kiln_mm
  cfg.geometry.kiln.d_kiln_mm = 700.0
  cfg.geometry.kiln.h_drum_mm = 595.0
  cfg.geometry.kiln.t_lid_mm = 75.0
  cfg.geometry.kiln.h_kiln_base_mm = 210.0

  # Bung geometry.
  cfg.geometry.bung.bung_angle_deg = 0.0
  cfg.geometry.bung.r_bung_mm = 415.0
  cfg.geometry.bung.z_bung_offset_mm = 540.0
  cfg.geometry.bung.bung_captured = point.bung_captured

  # SSR enclosure geometry.
  cfg.geometry.ssr.ssr_angle_deg = -45.0
  cfg.geometry.ssr.z_ssr_offset_mm = 100.0
  cfg.geometry.ssr.w_ssr_mm = 190.0
  cfg.geometry.ssr.d_ssr_mm = 146.5
  cfg.geometry.ssr.h_ssr_mm = 430.0

  # Floor stack geometry (40 mm screed from recent 40screed baseline).
  cfg.geometry.floor.d_reflector_mm = point.d_reflector_mm
  cfg.geometry.floor.t_reflector_calsil_mm = point.t_reflector_calsil_mm
  cfg.geometry.floor.g_floor_reflector_mm = point.g_floor_reflector_mm
  cfg.geometry.floor.w_floor_pad_mm = point.w_floor_pad_mm
  cfg.geometry.floor.t_floor_pad_mm = point.t_floor_pad_mm
  cfg.geometry.floor.t_screed_mm = 40.0
  cfg.geometry.floor.t_vip_mm = 20.0

  # Exhaust hood geometry centered dynamically over the kiln axis.
  x_kiln_center_mm = (
      cfg.geometry.room.chimney.d_chimney_mm
      + cfg.geometry.shield.t_shield_mm
      + cfg.geometry.kiln.s_kiln_left_mm
      + 0.5 * cfg.geometry.kiln.d_kiln_mm
  )
  y_kiln_center_mm = (
      cfg.geometry.room.chimney.d_chimney_mm
      + cfg.geometry.shield.t_shield_mm
      + cfg.geometry.kiln.s_kiln_back_mm
      + 0.5 * cfg.geometry.kiln.d_kiln_mm
  )
  cfg.geometry.hood.d_hood_inlet_mm = point.d_hood_inlet_mm
  cfg.geometry.hood.d_hood_duct_mm = 150.0
  cfg.geometry.hood.z_hood_drop_mm = 400.0
  cfg.geometry.hood.x_hood_mm = x_kiln_center_mm
  cfg.geometry.hood.y_hood_mm = y_kiln_center_mm

  # Thermal sources, emissivities, and conductivities.
  cfg.thermal.sources.t_internal_c = 1280.0
  cfg.thermal.sources.t_ext_c = 20.0
  cfg.thermal.sources.u_kiln_side = point.u_kiln_side
  cfg.thermal.sources.u_kiln_top = 2.00
  cfg.thermal.sources.u_kiln_bot = point.u_kiln_side
  cfg.thermal.sources.q_bung_open_w = 110.0
  cfg.thermal.sources.q_ssr_internal_w = 18.8
  cfg.thermal.sources.exhaust_fan_cfm = point.exhaust_fan_cfm

  cfg.thermal.emissivities.eps_kiln_side = 0.60
  cfg.thermal.emissivities.eps_kiln_top = 0.85
  cfg.thermal.emissivities.eps_kiln_bot = 0.15
  cfg.thermal.emissivities.eps_shield = 0.90
  cfg.thermal.emissivities.eps_gypsum = 0.90
  cfg.thermal.emissivities.eps_aac = 0.88
  cfg.thermal.emissivities.eps_reflector_top = 0.15
  cfg.thermal.emissivities.eps_ssr = 0.60

  cfg.thermal.conductivities.k_shield = 0.175
  cfg.thermal.conductivities.k_gypsum = 0.24
  cfg.thermal.conductivities.k_aac = 0.13
  cfg.thermal.conductivities.k_pad = 1.40
  cfg.thermal.conductivities.k_screed = 1.20
  cfg.thermal.conductivities.k_vip = 0.007
  cfg.thermal.conductivities.u_roof = 0.18
  cfg.thermal.conductivities.u_glass_door = 1.40

  # Discretization grid.
  cfg.grid.dx_wall_mm = 50.0
  cfg.grid.dz_wall_mm = 50.0
  cfg.grid.n_theta_kiln = 24
  cfg.grid.n_z_kiln = 6

  return cfg


def write_config_textproto(
    cfg: kiln_sim_pb2.SimulationConfig, config_path: pathlib.Path
) -> None:
  """Serializes a SimulationConfig protobuf to a human-readable .textproto file.

  Explicitly writes all fields (including zero-valued proto3 scalars) so files
  can be inspected and grepped directly.

  Args:
    cfg: Populated SimulationConfig protobuf message.
    config_path: Destination .textproto path.
  """
  bung_cap_str = "true" if cfg.geometry.bung.bung_captured else "false"
  seal_sides_str = (
      "true" if cfg.geometry.room.chimney.seal_chimney_sides else "false"
  )
  content = f"""# proto-file: proto/kiln_sim.proto
# proto-message: kiln_sim.SimulationConfig

experiment_name: "{cfg.experiment_name}"

geometry {{
  shield {{
    w_shield_left_mm: {cfg.geometry.shield.w_shield_left_mm:.1f}
    w_shield_back_mm: {cfg.geometry.shield.w_shield_back_mm:.1f}
    g_floor_mm: {cfg.geometry.shield.g_floor_mm:.1f}
    g_ceil_mm: {cfg.geometry.shield.g_ceil_mm:.1f}
    t_shield_mm: {cfg.geometry.shield.t_shield_mm:.1f}
  }}

  room {{
    h_wall_mm: {cfg.geometry.room.h_wall_mm:.1f}
    t_gypsum_mm: {cfg.geometry.room.t_gypsum_mm:.1f}
    t_aac_mm: {cfg.geometry.room.t_aac_mm:.1f}
    l_room_mm: {cfg.geometry.room.l_room_mm:.1f}
    w_room_mm: {cfg.geometry.room.w_room_mm:.1f}
    w_glass_door_mm: {cfg.geometry.room.w_glass_door_mm:.1f}
    w_door_open_mm: {cfg.geometry.room.w_door_open_mm:.1f}
    h_door_mm: {cfg.geometry.room.h_door_mm:.1f}

    chimney {{
      d_chimney_mm: {cfg.geometry.room.chimney.d_chimney_mm:.1f}
      chimney_blockage_fraction: {cfg.geometry.room.chimney.chimney_blockage_fraction:.2f}
      seal_chimney_sides: {seal_sides_str}
    }}
  }}

  kiln {{
    s_kiln_left_mm: {cfg.geometry.kiln.s_kiln_left_mm:.1f}
    s_kiln_back_mm: {cfg.geometry.kiln.s_kiln_back_mm:.1f}
    d_kiln_mm: {cfg.geometry.kiln.d_kiln_mm:.1f}
    h_drum_mm: {cfg.geometry.kiln.h_drum_mm:.1f}
    t_lid_mm: {cfg.geometry.kiln.t_lid_mm:.1f}
    h_kiln_base_mm: {cfg.geometry.kiln.h_kiln_base_mm:.1f}
  }}

  bung {{
    bung_angle_deg: {cfg.geometry.bung.bung_angle_deg:.1f}
    r_bung_mm: {cfg.geometry.bung.r_bung_mm:.1f}
    z_bung_offset_mm: {cfg.geometry.bung.z_bung_offset_mm:.1f}
    bung_captured: {bung_cap_str}
  }}

  ssr {{
    ssr_angle_deg: {cfg.geometry.ssr.ssr_angle_deg:.1f}
    z_ssr_offset_mm: {cfg.geometry.ssr.z_ssr_offset_mm:.1f}
    w_ssr_mm: {cfg.geometry.ssr.w_ssr_mm:.1f}
    d_ssr_mm: {cfg.geometry.ssr.d_ssr_mm:.1f}
    h_ssr_mm: {cfg.geometry.ssr.h_ssr_mm:.1f}
  }}

  floor {{
    d_reflector_mm: {cfg.geometry.floor.d_reflector_mm:.1f}
    t_reflector_calsil_mm: {cfg.geometry.floor.t_reflector_calsil_mm:.1f}
    g_floor_reflector_mm: {cfg.geometry.floor.g_floor_reflector_mm:.1f}
    w_floor_pad_mm: {cfg.geometry.floor.w_floor_pad_mm:.1f}
    t_floor_pad_mm: {cfg.geometry.floor.t_floor_pad_mm:.1f}
    t_screed_mm: {cfg.geometry.floor.t_screed_mm:.1f}
    t_vip_mm: {cfg.geometry.floor.t_vip_mm:.1f}
  }}

  hood {{
    d_hood_inlet_mm: {cfg.geometry.hood.d_hood_inlet_mm:.1f}
    d_hood_duct_mm: {cfg.geometry.hood.d_hood_duct_mm:.1f}
    z_hood_drop_mm: {cfg.geometry.hood.z_hood_drop_mm:.1f}
    x_hood_mm: {cfg.geometry.hood.x_hood_mm:.1f}
    y_hood_mm: {cfg.geometry.hood.y_hood_mm:.1f}
  }}
}}

thermal {{
  sources {{
    t_internal_c: {cfg.thermal.sources.t_internal_c:.1f}
    t_ext_c: {cfg.thermal.sources.t_ext_c:.1f}
    u_kiln_side: {cfg.thermal.sources.u_kiln_side:.2f}
    u_kiln_top: {cfg.thermal.sources.u_kiln_top:.2f}
    u_kiln_bot: {cfg.thermal.sources.u_kiln_bot:.2f}
    q_bung_open_w: {cfg.thermal.sources.q_bung_open_w:.1f}
    q_ssr_internal_w: {cfg.thermal.sources.q_ssr_internal_w:.1f}
    exhaust_fan_cfm: {cfg.thermal.sources.exhaust_fan_cfm:.1f}
  }}

  emissivities {{
    eps_kiln_side: {cfg.thermal.emissivities.eps_kiln_side:.2f}
    eps_kiln_top: {cfg.thermal.emissivities.eps_kiln_top:.2f}
    eps_kiln_bot: {cfg.thermal.emissivities.eps_kiln_bot:.2f}
    eps_shield: {cfg.thermal.emissivities.eps_shield:.2f}
    eps_gypsum: {cfg.thermal.emissivities.eps_gypsum:.2f}
    eps_aac: {cfg.thermal.emissivities.eps_aac:.2f}
    eps_reflector_top: {cfg.thermal.emissivities.eps_reflector_top:.2f}
    eps_ssr: {cfg.thermal.emissivities.eps_ssr:.2f}
  }}

  conductivities {{
    k_shield: {cfg.thermal.conductivities.k_shield:.3f}
    k_gypsum: {cfg.thermal.conductivities.k_gypsum:.2f}
    k_aac: {cfg.thermal.conductivities.k_aac:.2f}
    k_pad: {cfg.thermal.conductivities.k_pad:.2f}
    k_screed: {cfg.thermal.conductivities.k_screed:.2f}
    k_vip: {cfg.thermal.conductivities.k_vip:.3f}
    u_roof: {cfg.thermal.conductivities.u_roof:.2f}
    u_glass_door: {cfg.thermal.conductivities.u_glass_door:.2f}
  }}
}}

grid {{
  dx_wall_mm: {cfg.grid.dx_wall_mm:.1f}
  dz_wall_mm: {cfg.grid.dz_wall_mm:.1f}
  n_theta_kiln: {cfg.grid.n_theta_kiln}
  n_z_kiln: {cfg.grid.n_z_kiln}
}}
"""
  config_path.write_text(content, encoding="utf-8")


def _run_geometry_group(
    config_paths: Sequence[str], output_dir_str: str
) -> tuple[int, int, list[FailedRun]]:
  """Solves all configurations sharing identical mesh and view-factor geometry.

  Each configuration is isolated in its own try/except block so a failure in
  one run never aborts the remaining runs.

  Args:
    config_paths: Sequence of .textproto file paths sharing the same geometry.
    output_dir_str: Destination directory for .out.textproto files.

  Returns:
    Tuple of (total_runs_attempted, succeeded_runs_count, failed_runs_list).
  """
  output_dir = pathlib.Path(output_dir_str)
  mesh = None
  vf_sys = None
  geometry_error: str | None = None
  attempted = 0
  succeeded_count = 0
  failed_runs: list[FailedRun] = []

  for config_path_str in config_paths:
    attempted += 1
    config_path = pathlib.Path(config_path_str)
    exp_name = config_path.stem

    try:
      config_pb, config_md5 = io_manager.load_simulation_config(config_path)
      exp_name = config_pb.experiment_name or exp_name
      logging.info("Starting experiment: %s", exp_name)
      params = params_lib.from_proto(config_pb)

      if geometry_error is not None:
        failed_runs.append(
            FailedRun(
                experiment_name=exp_name,
                config_path=config_path_str,
                reason=f"Geometry/view-factor build failed: {geometry_error}",
            )
        )
        continue

      if mesh is None or vf_sys is None:
        try:
          mesh = mesh_lib.build_mesh(params)
          vf_sys = vf_lib.compute_view_factors(mesh, params)
        except Exception as exc:  # pylint: disable=broad-exception-caught
          geometry_error = f"{type(exc).__name__}: {exc}"
          failed_runs.append(
              FailedRun(
                  experiment_name=exp_name,
                  config_path=config_path_str,
                  reason=f"Geometry/view-factor build failed: {geometry_error}",
              )
          )
          continue

      solution = solver_lib.solve_thermal_system(
          mesh=mesh, vf_sys=vf_sys, params=params
      )
      timestamp_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
          "%Y-%m-%dT%H:%M:%SZ"
      )
      output_pb = metrics_lib.build_simulation_output(
          config_pb=config_pb,
          input_config_path=config_path_str,
          input_config_md5=config_md5,
          timestamp_utc=timestamp_utc,
          params=params,
          mesh=mesh,
          vf_sys=vf_sys,
          solution=solution,
      )

      out_filename = f"{exp_name}.out.textproto"
      io_manager.write_simulation_output(
          output_pb=output_pb,
          input_config_path=config_path,
          explicit_output_file=output_dir / out_filename,
      )

      if solution.converged:
        succeeded_count += 1
      else:
        dt_room_val = getattr(solution, "dt_room_change_k", 0.0)
        failed_runs.append(
            FailedRun(
                experiment_name=exp_name,
                config_path=config_path_str,
                reason=(
                    "Solver did not converge "
                    f"(max_energy_residual={solution.max_energy_residual_w_m2:.3f} W/m^2, "
                    f"dt_room_change={dt_room_val:.3f} K)"
                ),
            )
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failed_runs.append(
          FailedRun(
              experiment_name=exp_name,
              config_path=config_path_str,
              reason=f"{type(exc).__name__}: {exc}",
          )
      )

  return attempted, succeeded_count, failed_runs


def _summarize_failures(failed_runs: Sequence[FailedRun], total_runs: int) -> None:
  """Logs a summary of any failed or non-converged runs at the end of execution."""
  if not failed_runs:
    logging.info("Failure summary: 0/%d runs failed.", total_runs)
    return

  logging.warning(
      "Failure summary: %d/%d runs failed or did not converge:",
      len(failed_runs),
      total_runs,
  )
  for failure in failed_runs:
    logging.warning(
        "  FAILED: %s (%s) -> %s",
        failure.experiment_name,
        failure.config_path,
        failure.reason,
    )


SWEEP_PARAM_COLUMNS: tuple[str, ...] = (
    "s_kiln_mm",
    "u_kiln_side",
    "w_floor_pad_mm",
    "t_floor_pad_mm",
    "w_door_open_mm",
    "exhaust_fan_cfm",
    "d_reflector_mm",
    "t_reflector_calsil_mm",
    "g_floor_reflector_mm",
    "d_chimney_mm",
    "g_chimney_mm",
    "w_shield_mm",
    "d_hood_inlet_mm",
    "bung_captured",
)

METRIC_COLUMNS: tuple[str, ...] = tuple(
    field.name
    for field in kiln_sim_pb2.SimulationOutput.SafetyMetrics.DESCRIPTOR.fields
    if not field.name.endswith("_safe")
) + ("t_kiln_skin_max_c",)

_KILN_SIDE_TEMP_RE = re.compile(
    rb"surface_type:\s*SURFACE_TYPE_KILN_SIDE\n(?:  [^\n]+\n)*?  temp_front_c:\s*([-0-9.eE+]+)"
)


def _extract_kiln_skin_max_c(
    out_path: pathlib.Path, expected_kiln_patches: int
) -> float:
  """Extracts the peak KILN_SIDE front-face temperature (C) from `out_path`."""
  file_size = out_path.stat().st_size
  tail_bytes = max(65536, expected_kiln_patches * 512)
  with out_path.open("rb") as f:
    if file_size > tail_bytes:
      f.seek(file_size - tail_bytes)
    chunk = f.read()
  temps = [float(m.group(1)) for m in _KILN_SIDE_TEMP_RE.finditer(chunk)]
  if len(temps) < expected_kiln_patches and file_size > tail_bytes:
    chunk = out_path.read_bytes()
    temps = [float(m.group(1)) for m in _KILN_SIDE_TEMP_RE.finditer(chunk)]
  return max(temps) if temps else 0.0


def expected_patch_count(params: params_lib.SimParams) -> int:
  """Computes the exact number of discretized patches built for `params`."""
  n_cols_left = max(1, int(round(params.w_shield_left / params.dx_wall)))
  n_cols_back = max(1, int(round(params.w_shield_back / params.dx_wall)))
  n_rows_shield = max(1, int(round(params.h_shield / params.dz_wall)))
  n_rows_wall = max(1, int(round(params.h_wall / params.dz_wall)))

  gyp_band_width = 1.0
  x_gyp_start = params.x_shield_left_end
  x_gyp_end = min(params.l_room, x_gyp_start + gyp_band_width)
  w_gyp_left = max(params.dx_wall, x_gyp_end - x_gyp_start)
  n_cols_gyp_left = max(1, int(round(w_gyp_left / params.dx_wall)))

  y_gyp_start = params.y_shield_back_end
  y_gyp_end = min(params.w_room, y_gyp_start + gyp_band_width)
  w_gyp_back = max(params.dx_wall, y_gyp_end - y_gyp_start)
  n_cols_gyp_back = max(1, int(round(w_gyp_back / params.dx_wall)))

  return (
      (n_cols_left + n_cols_back) * n_rows_shield
      + (n_cols_gyp_left + n_cols_gyp_back) * n_rows_wall
      + params.n_theta_kiln * params.n_z_kiln
      + 6
  )


def is_well_formed_output(
    out_path: pathlib.Path, params: params_lib.SimParams
) -> bool:
  """Returns True if `out_path` is a converged output with all patch sections."""
  if not out_path.is_file():
    return False
  try:
    data = out_path.read_bytes()
    idx = data.find(b"\npatches {\n")
    if idx == -1 or not data.rstrip().endswith(b"}"):
      return False
    header_text = data[:idx].decode("utf-8")
    output_pb = kiln_sim_pb2.SimulationOutput()
    text_format.Parse(header_text, output_pb)
    if not output_pb.provenance.converged or not output_pb.HasField("metrics"):
      return False
    patch_count = data.count(b"\npatches {\n", idx)
    return patch_count == expected_patch_count(params)
  except Exception:  # pylint: disable=broad-exception-caught
    return False


def filter_pending_points(
    points: Sequence[SweepPoint],
    output_dir: pathlib.Path,
    params_override: Optional[params_lib.SimParams] = None,
) -> list[SweepPoint]:
  """Filters out sweep points that already have a well-formed output file."""
  pending: list[SweepPoint] = []
  for point in points:
    out_path = output_dir / f"{point.experiment_name}.out.textproto"
    params = (
        params_override
        if params_override is not None
        else params_lib.from_proto(build_config_proto(point))
    )
    if not is_well_formed_output(out_path, params):
      pending.append(point)
  return pending


def _parse_output_header(
    out_path: pathlib.Path,
) -> kiln_sim_pb2.SimulationOutput:
  """Parses SimulationOutput header sections (excluding patches) for speed."""
  with out_path.open("r", encoding="utf-8") as f:
    lines: list[str] = []
    for line in f:
      if line.startswith("patches {"):
        break
      lines.append(line)
  header_text = "".join(lines)
  output_pb = kiln_sim_pb2.SimulationOutput()
  text_format.Parse(header_text, output_pb)
  return output_pb


def export_sweep_csv(
    output_dir: pathlib.Path, csv_path: pathlib.Path
) -> int:
  """Exports a CSV with one column per sweep param and non-*_safe metric.

  Args:
    output_dir: Directory containing `.out.textproto` result files.
    csv_path: Destination CSV file path.

  Returns:
    Number of simulation result rows written to `csv_path`.
  """
  out_files = sorted(output_dir.glob("*.out.textproto"))
  csv_path.parent.mkdir(parents=True, exist_ok=True)
  headers = list(SWEEP_PARAM_COLUMNS) + list(METRIC_COLUMNS)

  rows_written = 0
  with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for out_path in out_files:
      output_pb = _parse_output_header(out_path)
      cfg = output_pb.input_config_snapshot
      metrics = output_pb.metrics
      param_values = [
          cfg.geometry.kiln.s_kiln_left_mm,
          cfg.thermal.sources.u_kiln_side,
          cfg.geometry.floor.w_floor_pad_mm,
          cfg.geometry.floor.t_floor_pad_mm,
          cfg.geometry.room.w_door_open_mm,
          cfg.thermal.sources.exhaust_fan_cfm,
          cfg.geometry.floor.d_reflector_mm,
          cfg.geometry.floor.t_reflector_calsil_mm,
          cfg.geometry.floor.g_floor_reflector_mm,
          cfg.geometry.room.chimney.d_chimney_mm,
          cfg.geometry.shield.g_floor_mm,
          cfg.geometry.shield.w_shield_left_mm,
          cfg.geometry.hood.d_hood_inlet_mm,
          cfg.geometry.bung.bung_captured,
      ]
      n_kiln_patches = max(1, cfg.grid.n_theta_kiln * cfg.grid.n_z_kiln)
      metric_values = [
          _extract_kiln_skin_max_c(out_path, n_kiln_patches)
          if col == "t_kiln_skin_max_c"
          else getattr(metrics, col)
          for col in METRIC_COLUMNS
      ]
      writer.writerow(param_values + metric_values)
      rows_written += 1

  logging.info("Wrote %d rows to %s", rows_written, csv_path)
  return rows_written


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected positional arguments: {argv[1:]}")

  config_dir = pathlib.Path(_CONFIG_DIR.value)
  output_dir = pathlib.Path(_OUTPUT_DIR.value)
  csv_path = pathlib.Path(_CSV_OUTPUT.value)

  if _DRY_RUN.value:
    sweep_points = enumerate_sweep_points()
    if _QUICK.value:
      max_workers = max(1, _JOBS.value)
      sweep_points = reservoir_sample_points(sweep_points, 2 * max_workers)
    pending_points = filter_pending_points(sweep_points, output_dir)
    for point in pending_points:
      print(f"{point.experiment_name}.out.textproto")
    return

  io_manager.configure_absl_logging(binary_name="run_sweep")

  if _CSV_ONLY.value:
    export_sweep_csv(output_dir, csv_path)
    return
  config_dir.mkdir(parents=True, exist_ok=True)
  output_dir.mkdir(parents=True, exist_ok=True)

  max_workers = max(1, _JOBS.value)
  all_sweep_points = enumerate_sweep_points()
  if _QUICK.value:
    sample_size = 2 * max_workers
    total_available = len(all_sweep_points)
    all_sweep_points = reservoir_sample_points(all_sweep_points, sample_size)
    logging.info(
        "--quick enabled: reservoir-sampled %d/%d configurations "
        "(2 * max_workers=%d).",
        len(all_sweep_points),
        total_available,
        max_workers,
    )

  sweep_points = filter_pending_points(all_sweep_points, output_dir)
  skipped_count = len(all_sweep_points) - len(sweep_points)
  if skipped_count > 0:
    logging.info(
        "Skipping %d/%d runs that already have well-formed outputs in %s.",
        skipped_count,
        len(all_sweep_points),
        output_dir,
    )

  if not sweep_points:
    logging.info("All %d runs are already complete.", len(all_sweep_points))
    export_sweep_csv(output_dir, csv_path)
    return

  logging.info(
      "Generating %d sweep configurations in %s...",
      len(sweep_points),
      config_dir,
  )

  geometry_groups: dict[tuple[float, ...], list[str]] = {}
  failed_runs: list[FailedRun] = []

  for point in sweep_points:
    config_path = config_dir / f"{point.experiment_name}.textproto"
    try:
      cfg = build_config_proto(point)
      params_lib.from_proto(cfg)
      write_config_textproto(cfg, config_path)
      geometry_groups.setdefault(point.geometry_cache_key, []).append(
          str(config_path)
      )
    except Exception as exc:  # pylint: disable=broad-exception-caught
      failed_runs.append(
          FailedRun(
              experiment_name=point.experiment_name,
              config_path=str(config_path),
              reason=f"Config validation/write failed - {type(exc).__name__}: {exc}",
          )
      )

  valid_configs_count = sum(len(paths) for paths in geometry_groups.values())
  logging.info(
      "Wrote %d/%d .textproto files across %d unique geometries.",
      valid_configs_count,
      len(sweep_points),
      len(geometry_groups),
  )

  if _CONFIGS_ONLY.value:
    _summarize_failures(failed_runs, len(sweep_points))
    logging.info("--configs_only specified; skipping simulation runs.")
    return

  t_start = time.perf_counter()
  total_attempted = len(failed_runs)
  total_succeeded = 0
  num_groups = len(geometry_groups)

  logging.info(
      "Running %d simulations (%d geometry groups) using %d worker processes...",
      valid_configs_count,
      num_groups,
      max_workers,
  )

  with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    future_to_paths = {
        executor.submit(_run_geometry_group, paths, str(output_dir)): paths
        for paths in geometry_groups.values()
    }
    for idx, future in enumerate(futures.as_completed(future_to_paths), start=1):
      paths = future_to_paths[future]
      try:
        attempted, succeeded, group_failures = future.result()
        total_attempted += attempted
        total_succeeded += succeeded
        failed_runs.extend(group_failures)
      except Exception as exc:  # pylint: disable=broad-exception-caught
        total_attempted += len(paths)
        for p in paths:
          failed_runs.append(
              FailedRun(
                  experiment_name=pathlib.Path(p).stem,
                  config_path=p,
                  reason=f"Worker process failure - {type(exc).__name__}: {exc}",
              )
          )

      if idx % 24 == 0 or idx == num_groups:
        elapsed = time.perf_counter() - t_start
        logging.info(
            "Progress: %d/%d geometry groups (%d/%d runs finished, %d succeeded, "
            "%d failed) in %.1f s",
            idx,
            num_groups,
            total_attempted,
            len(sweep_points),
            total_succeeded,
            len(failed_runs),
            elapsed,
        )

  total_elapsed = time.perf_counter() - t_start
  logging.info(
      "Sweep complete: %d/%d simulations succeeded in %.2f s. Outputs in %s",
      total_succeeded,
      len(sweep_points),
      total_elapsed,
      output_dir,
  )
  _summarize_failures(failed_runs, len(sweep_points))
  if total_succeeded > 0:
    export_sweep_csv(output_dir, csv_path)


if __name__ == "__main__":
  flags.FLAGS.set_default("log_dir", str(io_manager.DEFAULT_BASE_LOG_DIR))
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)
