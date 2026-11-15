"""Unit tests verifying safety threshold consistency in kiln_explorer.html."""

import json
import os
import pathlib
import re
import subprocess
import unittest

from src import constants


def _find_kiln_explorer_path() -> pathlib.Path:
  candidates = [pathlib.Path("kiln_explorer.html")]
  test_srcdir = os.environ.get("TEST_SRCDIR")
  if test_srcdir:
    for root in pathlib.Path(test_srcdir).iterdir():
      candidates.append(root / "kiln_explorer.html")
  for path in candidates:
    if path.is_file():
      return path
  raise FileNotFoundError("Could not locate kiln_explorer.html")


def _extract_explorer_script(html_text: str) -> str:
  match = re.search(
      r"<script>([\s\S]*?)</script>", html_text, flags=re.IGNORECASE
  )
  if not match:
    raise AssertionError("Could not locate <script> block in kiln_explorer.html")
  script = match.group(1)
  script = re.sub(
      r'^const SWEEP_DATA_B64 = ".*?";$',
      'const SWEEP_DATA_B64 = "";',
      script,
      count=1,
      flags=re.MULTILINE,
  )
  startup_tail = (
      "setupControls();\nsyncControlButtons();\nrenderElevation();"
  )
  if not script.strip().endswith(startup_tail):
    raise AssertionError("Unexpected startup footer in kiln_explorer.html")
  script = script[: script.rfind(startup_tail)]
  return script


def _run_explorer_js(scenarios: dict[str, dict[str, object]]) -> dict[str, object]:
  html_text = _find_kiln_explorer_path().read_text(encoding="utf-8")
  core_script = _extract_explorer_script(html_text)
  harness = f"""
const __elements = {{
  'elevation-svg': {{ innerHTML: '', setAttribute() {{}} }},
  'header-subtitle': {{ textContent: '' }},
  'drag-hint-msg': {{ textContent: '' }},
  'plume-summary': {{ textContent: '' }},
  'metric-list': {{ innerHTML: '' }},
}};
globalThis.document = {{
  getElementById(id) {{ return __elements[id] || null; }},
  querySelectorAll() {{ return []; }},
}};
globalThis.window = globalThis;
globalThis.__capturedSensors = null;
{core_script}
const __origRenderCallouts = renderCallouts;
renderCallouts = function(sensors, o) {{
  globalThis.__capturedSensors = sensors.map(s => ({{ id: s.id, label: s.label, safe: Boolean(s.safe) }}));
  return __origRenderCallouts(sensors, o);
}};
let __mockMetrics = null;
getMetricsForState = () => __mockMetrics;
function __parseSidebar() {{
  const html = __elements['metric-list'].innerHTML;
  const out = {{}};
  const re = /<div class="metric-name">([^<]+)<\\/div>[\\s\\S]*?<span class="metric-badge([^"]*)">/g;
  let m;
  while ((m = re.exec(html)) !== null) {{
    out[m[1]] = !m[2].includes('unsafe');
  }}
  return out;
}}
const __scenarios = {json.dumps(scenarios)};
const __results = {{ limits: LIMITS, scenarios: {{}} }};
for (const [name, metrics] of Object.entries(__scenarios)) {{
  __mockMetrics = metrics;
  const perView = {{}};
  for (const view of ['side', 'front', 'plan']) {{
    state.view = view;
    globalThis.__capturedSensors = null;
    renderElevation();
    const sensorMap = {{}};
    for (const s of globalThis.__capturedSensors) sensorMap[s.id] = s.safe;
    perView[view] = {{
      sensors: sensorMap,
      sidebar: __parseSidebar(),
    }};
  }}
  __results.scenarios[name] = perView;
}}
process.stdout.write(JSON.stringify(__results));
"""
  proc = subprocess.run(
      ["/usr/bin/node", "-e", harness],
      check=True,
      capture_output=True,
      text=True,
  )
  return json.loads(proc.stdout)


def _baseline_safe_metrics() -> dict[str, object]:
  return {
      "t_gypsum_max_left_c": 32.0,
      "t_gypsum_max_back_c": 31.0,
      "t_ssr_air_c": 29.0,
      "t_ssr_case_c": 36.0,
      "t_vip_under_reflector_c": 38.0,
      "t_floor_pad_under_reflector_c": 52.0,
      "t_floor_pad_edge_c": 41.0,
      "t_vip_outside_reflector_c": 36.0,
      "t_vinyl_edge_c": 34.0,
      "t_shield_max_c": 120.0,
      "t_aac_max_c": 90.0,
      "t_ceiling_max_c": 33.0,
      "t_room_upper_c": 30.0,
      "t_room_lower_c": 25.0,
      "z_stratification_layer_mm": 1800,
      "plume_capture_fraction": 99.5,
      "plume_fully_captured": True,
      "t_kiln_skin_max_c": 150.0,
      "t_exhaust_duct_c": 28.0,
  }


class KilnExplorerSafetyTest(unittest.TestCase):
  """Tests UI safety limits and flag consistency across views and sidebar."""

  def test_limits_and_safety_flags_consistent_across_all_views(self) -> None:
    safe = _baseline_safe_metrics()
    gyp_back_bad = dict(
        safe, t_gypsum_max_left_c=35.0, t_gypsum_max_back_c=42.0
    )
    ssr_case_bad = dict(safe, t_ssr_air_c=31.0, t_ssr_case_c=47.0)
    vip_bad = dict(
        safe,
        t_vip_under_reflector_c=55.0,
        t_vip_outside_reflector_c=55.0,
    )
    shield_bad = dict(safe, t_shield_max_c=260.0)
    aac_bad = dict(safe, t_aac_max_c=130.0)

    out = _run_explorer_js({
        "safe": safe,
        "gyp_back_bad": gyp_back_bad,
        "ssr_case_bad": ssr_case_bad,
        "vip_bad": vip_bad,
        "shield_bad": shield_bad,
        "aac_bad": aac_bad,
    })

    limits = out["limits"]
    self.assertAlmostEqual(limits["gypsum"], constants.LIMIT_GYPSUM_C)
    self.assertAlmostEqual(limits["ceiling"], constants.LIMIT_CEILING_C)
    self.assertAlmostEqual(limits["ssr_air"], constants.LIMIT_SSR_AIR_C)
    self.assertAlmostEqual(limits["ssr_case"], constants.LIMIT_SSR_CASE_C)
    self.assertAlmostEqual(limits["vip"], constants.LIMIT_VIP_C)
    self.assertAlmostEqual(limits["vinyl"], constants.LIMIT_VINYL_C)
    self.assertAlmostEqual(limits["shield"], constants.LIMIT_SHIELD_C)
    self.assertAlmostEqual(limits["aac"], constants.LIMIT_AAC_C)

    scenarios = out["scenarios"]

    # 1. Baseline safe: every view sensor and sidebar badge is safe (including
    # t_shield_max_c=120 C < 250 C and t_aac_max_c=90 C < 120 C).
    for view in ("side", "front", "plan"):
      v_safe = scenarios["safe"][view]
      for sid, is_safe in v_safe["sensors"].items():
        self.assertTrue(
            is_safe, f"Expected sensor {sid} safe in {view} view"
        )
      for name, is_ok in v_safe["sidebar"].items():
        self.assertTrue(
            is_ok, f"Expected sidebar row {name} safe in {view} view"
        )

    # 2. Gypsum back exceedance: front gyp, plan gyp_back, and sidebar row fail.
    for view in ("side", "front", "plan"):
      v_gyp = scenarios["gyp_back_bad"][view]
      self.assertFalse(v_gyp["sidebar"]["Gypsum Max (Left / Back)"])
    self.assertFalse(scenarios["gyp_back_bad"]["front"]["sensors"]["gyp"])
    self.assertFalse(scenarios["gyp_back_bad"]["plan"]["sensors"]["gyp_back"])
    self.assertTrue(scenarios["gyp_back_bad"]["plan"]["sensors"]["gyp_left"])

    # 3. SSR case exceedance: ssr sensor and sidebar fail in every view.
    for view in ("side", "front", "plan"):
      v_ssr = scenarios["ssr_case_bad"][view]
      self.assertFalse(v_ssr["sensors"]["ssr"])
      self.assertFalse(v_ssr["sidebar"]["SSR Air / Casing"])

    # 4. VIP exceedance: pad/vip sensors and sidebar rows fail in every view.
    for view in ("side", "front", "plan"):
      v_vip = scenarios["vip_bad"][view]
      self.assertFalse(v_vip["sensors"]["pad_under"])
      self.assertFalse(v_vip["sensors"]["pad_edge"])
      self.assertFalse(v_vip["sidebar"]["VIP Under Reflector"])
      self.assertFalse(v_vip["sidebar"]["Pad Under Reflector"])
      self.assertFalse(v_vip["sidebar"]["VIP Outside Reflector"])
      self.assertFalse(v_vip["sidebar"]["Pad Outside Reflector"])
      if view in ("side", "front"):
        self.assertFalse(v_vip["sensors"]["vip_under"])
        self.assertFalse(v_vip["sensors"]["vip_out"])

    # 5. Shield exceedance (> 250 C) and AAC exceedance (> 120 C).
    for view in ("side", "front", "plan"):
      self.assertFalse(scenarios["shield_bad"][view]["sensors"]["shd"])
      self.assertFalse(
          scenarios["shield_bad"][view]["sidebar"]["Calsil Shield / AAC Max"]
      )
      self.assertFalse(scenarios["aac_bad"][view]["sensors"]["aac"])
      self.assertFalse(
          scenarios["aac_bad"][view]["sidebar"]["Calsil Shield / AAC Max"]
      )


if __name__ == "__main__":
  unittest.main()

