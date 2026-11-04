"""Abseil CLI entry point for running a single kiln thermal simulation."""

from collections.abc import Sequence
import pathlib

from absl import app
from absl import flags
from absl import logging
from src import io_manager
from src import mesh as mesh_lib
from src import metrics as metrics_lib
from src import params as params_lib
from src import solver as solver_lib
from src import view_factors as vf_lib

FLAGS = flags.FLAGS

_CONFIG = flags.DEFINE_string(
    "config",
    None,
    "Path to the input SimulationConfig .textproto file.",
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "results",
    "Directory where the timestamped .out.textproto file will be written when "
    "--output_file is not explicitly provided.",
)
_OUTPUT_FILE = flags.DEFINE_string(
    "output_file",
    None,
    "Optional explicit file path for the SimulationOutput .out.textproto file.",
)


def main(argv: Sequence[str]) -> None:
  """Executes a single simulation run and writes the provenance-linked output."""
  if len(argv) > 1:
    raise app.UsageError(f"Unexpected command-line arguments: {argv[1:]}")

  io_manager.configure_absl_logging(binary_name="main")

  config_path = pathlib.Path(_CONFIG.value)
  config_pb, md5_hex = io_manager.load_simulation_config(config_path)

  params = params_lib.from_proto(config_pb)
  mesh = mesh_lib.build_mesh(params)
  logging.info("Constructed SurfaceMesh with %d patches.", mesh.num_patches)

  vf_sys = vf_lib.compute_view_factors(mesh, params)
  solution = solver_lib.solve_thermal_system(mesh, vf_sys, params)

  t_kiln_front_c = metrics_lib.get_kiln_front_face_temp_c(
      mesh=mesh, solution=solution, params=params
  )
  t_shield_behind_kiln_c = metrics_lib.get_shield_temp_behind_kiln_c(
      mesh=mesh, solution=solution, params=params
  )

  iso_ts, compact_ts = io_manager.format_utc_timestamp()
  output_pb = metrics_lib.build_simulation_output(
      config_pb=config_pb,
      input_config_path=str(config_path),
      input_config_md5=md5_hex,
      timestamp_utc=iso_ts,
      params=params,
      mesh=mesh,
      vf_sys=vf_sys,
      solution=solution,
  )

  explicit_out = (
      pathlib.Path(_OUTPUT_FILE.value) if _OUTPUT_FILE.value else None
  )
  out_dir = pathlib.Path(_OUTPUT_DIR.value) if _OUTPUT_DIR.value else None
  written_path = io_manager.write_simulation_output(
      output_pb=output_pb,
      input_config_path=config_path,
      output_dir=out_dir,
      explicit_output_file=explicit_out,
      compact_timestamp=compact_ts,
  )

  m = output_pb.metrics
  logging.info(
      "Simulation complete (converged=%s, md5=%s):\n"
      "  Kiln Front Face Temp                      : %.1f C (delta_T_skin=%.1f C)\n"
      "  Shield Behind Kiln (+x)                   : %.1f C (shield_max=%.1f C)\n"
      "  Gypsum Max (Left / Back)                  : %.1f C / %.1f C (safe=%s)\n"
      "  Ceiling Max Temp                          : %.1f C (safe=%s)\n"
      "  SSR Temp (Air / Casing)                   : %.1f C / %.1f C (safe=%s)\n"
      "  Floor Under Reflector (VIP / Pad)         : %.1f C / %.1f C (safe=%s)\n"
      "  Floor Outside Reflector (VIP / Pad / Vinyl): %.1f C / %.1f C / %.1f C (safe=%s)\n"
      "  Plume Capture Fraction                    : %.1f%% (fully_captured=%s)\n"
      "  Chimney Exit Max                          : %.1f C @ %.2f m/s (captured=%.1f%%)\n"
      "  Room Air (Upper / Lower)                  : %.1f C / %.1f C\n"
      "  Output File                               : %s",
      output_pb.provenance.converged,
      md5_hex,
      t_kiln_front_c,
      m.delta_t_kiln_skin_c,
      t_shield_behind_kiln_c,
      m.t_shield_max_c,
      m.t_gypsum_max_left_c,
      m.t_gypsum_max_back_c,
      m.gypsum_safe,
      m.t_ceiling_max_c,
      m.ceiling_safe,
      m.t_ssr_air_c,
      m.t_ssr_case_c,
      m.ssr_safe,
      m.t_vip_under_reflector_c,
      m.t_floor_pad_under_reflector_c,
      m.floor_under_reflector_safe,
      m.t_vip_outside_reflector_c,
      m.t_floor_pad_edge_c,
      m.t_vinyl_edge_c,
      m.floor_outside_reflector_safe,
      m.plume_capture_fraction * 100.0,
      m.plume_fully_captured,
      m.t_chimney_exit_max_c,
      m.v_chimney_max_m_s,
      m.chimney_capture_fraction * 100.0,
      m.t_room_upper_c,
      m.t_room_lower_c,
      written_path,
  )


if __name__ == "__main__":
  flags.FLAGS.set_default("log_dir", str(io_manager.DEFAULT_BASE_LOG_DIR))
  flags.FLAGS.set_default("alsologtostderr", True)
  app.run(main)