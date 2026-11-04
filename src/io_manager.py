"""Configuration loading, MD5 provenance hashing, and output serialization."""

import datetime
import hashlib
import pathlib
from typing import Optional

from absl import flags
from absl import logging
from google.protobuf import text_format
from proto import kiln_sim_pb2

DEFAULT_BASE_LOG_DIR: pathlib.Path = pathlib.Path("/usr/local/google/tmp")


def configure_absl_logging(
    binary_name: str,
    base_log_dir: Optional[pathlib.Path] = None,
    dt: Optional[datetime.datetime] = None,
) -> pathlib.Path:
  """Creates a unique per-run subdirectory and starts Abseil file logging.

  Subdirectory name uses `<yyyymmdd-hhmmss.s>_<binary_name>` (with tenths of a
  second) under `base_log_dir` (defaulting to `/usr/local/google/tmp`).

  Args:
    binary_name: Name of the executable (e.g., `"main"` or `"run_sweep"`).
    base_log_dir: Base directory for logs. If None, uses `FLAGS.log_dir` if set,
      otherwise `DEFAULT_BASE_LOG_DIR`.
    dt: Optional timestamp for directory naming. If None, uses current local
      time.

  Returns:
    The created per-run logging subdirectory Path.
  """
  if base_log_dir is None:
    flag_log_dir = flags.FLAGS["log_dir"].value if "log_dir" in flags.FLAGS else ""
    base_log_dir = (
        pathlib.Path(flag_log_dir) if flag_log_dir else DEFAULT_BASE_LOG_DIR
    )

  if dt is None:
    dt = datetime.datetime.now()

  tenths = dt.microsecond // 100000
  timestamp_str = f"{dt.strftime('%Y%m%d-%H%M%S')}.{tenths}"
  run_log_dir = base_log_dir / f"{timestamp_str}_{binary_name}"
  run_log_dir.mkdir(parents=True, exist_ok=True)

  if not flags.FLAGS.is_parsed():
    flags.FLAGS.mark_as_parsed()
  if "log_dir" in flags.FLAGS:
    flags.FLAGS["log_dir"].value = str(run_log_dir)
  logging.get_absl_handler().use_absl_log_file(
      program_name=binary_name, log_dir=str(run_log_dir)
  )
  logging.info("Abseil file logging initialized in %s", run_log_dir)
  return run_log_dir


def load_simulation_config(
    config_path: pathlib.Path,
) -> tuple[kiln_sim_pb2.SimulationConfig, str]:
  """Reads a .textproto file, computes its raw-byte MD5, and parses the config.

  Reads the file bytes into memory once so that the computed MD5 hash matches
  the exact bytes parsed by protobuf and the external `md5sum` CLI utility.

  Args:
    config_path: Path to the input .textproto configuration file.

  Returns:
    A tuple of (parsed SimulationConfig protobuf message, 32-character lowercase
    hexadecimal MD5 digest string).

  Raises:
    FileNotFoundError: If config_path does not exist.
    text_format.ParseError: If the file is not valid SimulationConfig textproto.
  """
  resolved_path = config_path.resolve()
  raw_bytes = resolved_path.read_bytes()
  md5_hex = hashlib.md5(raw_bytes).hexdigest()

  config_pb = kiln_sim_pb2.SimulationConfig()
  text_format.Parse(raw_bytes.decode("utf-8"), config_pb)

  logging.info(
      "Loaded config '%s' from %s (md5=%s)",
      config_pb.experiment_name,
      resolved_path,
      md5_hex,
  )
  return config_pb, md5_hex


def format_utc_timestamp(
    dt: Optional[datetime.datetime] = None,
) -> tuple[str, str]:
  """Formats a UTC datetime into ISO-8601 and compact filename strings.

  Args:
    dt: Optional UTC datetime. If None, uses current UTC time.

  Returns:
    A tuple of (iso_8601_str, filename_compact_str), e.g.,
    ("2026-11-01T19:58:00Z", "20261101T195800Z").
  """
  if dt is None:
    dt = datetime.datetime.now(datetime.timezone.utc)
  iso_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
  compact_str = dt.strftime("%Y%m%dT%H%M%SZ")
  return iso_str, compact_str


def write_simulation_output(
    output_pb: kiln_sim_pb2.SimulationOutput,
    input_config_path: pathlib.Path,
    output_dir: Optional[pathlib.Path] = None,
    explicit_output_file: Optional[pathlib.Path] = None,
    compact_timestamp: Optional[str] = None,
) -> pathlib.Path:
  """Serializes a SimulationOutput message to a human-readable .textproto file.

  If explicit_output_file is provided, writes directly to that path. Otherwise,
  constructs a timestamped filename `<input_stem>_<YYYYMMDDTHHMMSSZ>.out.textproto`
  inside output_dir.

  Args:
    output_pb: Populated SimulationOutput protobuf message.
    input_config_path: Original input config path (used for stem naming).
    output_dir: Target directory when explicit_output_file is not set.
    explicit_output_file: Optional explicit file path for the output.
    compact_timestamp: Optional compact UTC timestamp string (`YYYYMMDDTHHMMSSZ`).

  Returns:
    The resolved Path of the written .out.textproto file.
  """
  if explicit_output_file is not None:
    target_path = explicit_output_file.resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
  else:
    if output_dir is None:
      output_dir = pathlib.Path.cwd() / "results"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if compact_timestamp is None:
      _, compact_timestamp = format_utc_timestamp()
    stem = input_config_path.stem
    filename = f"{stem}_{compact_timestamp}.out.textproto"
    target_path = output_dir / filename

  header_comment = (
      "# proto-file: proto/kiln_sim.proto\n"
      "# proto-message: kiln_sim.SimulationOutput\n\n"
  )
  serialized_text = header_comment + text_format.MessageToString(output_pb)

  temp_path = target_path.with_suffix(".tmp")
  temp_path.write_text(serialized_text, encoding="utf-8")
  temp_path.replace(target_path)

  logging.info("Wrote simulation output to %s", target_path)
  return target_path