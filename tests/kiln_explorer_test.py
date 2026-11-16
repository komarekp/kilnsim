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
      "loadStateFromUrl();\nsetupControls();\nsyncControlButtons();\n"
      "syncUrlFromState();\nrenderElevation();"
  )
  if not script.strip().endswith(startup_tail):
    raise AssertionError("Unexpected startup footer in kiln_explorer.html")
  return script[: script.rfind(startup_tail)]


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

  def test_url_parameter_parsing_and_button_click_url_updates(self) -> None:
    html_text = _find_kiln_explorer_path().read_text(encoding="utf-8")
    core_script = _extract_explorer_script(html_text)
    button_groups = {
        "ctrl-view": ["side", "front", "plan"],
        "ctrl-skiln": ["200", "300"],
        "ctrl-ukiln": ["0.8", "0.5", "0.3"],
        "ctrl-fan": ["350", "400", "450"],
        "ctrl-hood": ["600", "700", "800"],
        "ctrl-hdrop": ["150", "400", "600"],
        "ctrl-dref": ["0", "800"],
        "ctrl-gref": ["0", "20"],
        "ctrl-tref": ["0", "10"],
        "ctrl-tpad": ["0", "20"],
        "ctrl-door": ["0", "150"],
    }
    harness = f"""
const __elements = {{
  'elevation-svg': {{ innerHTML: '', setAttribute() {{}} }},
  'header-subtitle': {{ textContent: '' }},
  'drag-hint-msg': {{ textContent: '' }},
  'plume-summary': {{ textContent: '' }},
  'metric-list': {{ innerHTML: '' }},
}};
const __buttonSpec = {json.dumps(button_groups)};
const __buttonMap = {{}};
for (const [gid, vals] of Object.entries(__buttonSpec)) {{
  __buttonMap[gid] = vals.map(v => ({{
    dataset: {{ val: v }},
    disabled: false,
    _active: false,
    _handlers: {{}},
    classList: {{ toggle(cls, on) {{ if (cls === 'active') this._active = Boolean(on); }} }},
    addEventListener(ev, fn) {{ this._handlers[ev] = fn; }},
    click() {{ if (this._handlers.click) this._handlers.click(); }},
  }}));
}}
globalThis.document = {{
  getElementById(id) {{
    if (!__elements[id] && id.startsWith('lbl-')) {{
      __elements[id] = {{ textContent: '' }};
    }}
    return __elements[id] || null;
  }},
  querySelectorAll(sel) {{
    const m = /^#([a-z-]+) button$/.exec(sel);
    return m && __buttonMap[m[1]] ? __buttonMap[m[1]] : [];
  }},
}};
const __replaceCalls = [];
globalThis.window = globalThis;
globalThis.window.location = {{ pathname: '/kiln_explorer.html', search: '', hash: '' }};
globalThis.window.history = {{
  replaceState(_s, _t, url) {{
    __replaceCalls.push(url);
    const qIdx = url.indexOf('?');
    globalThis.window.location.search = qIdx >= 0 ? url.slice(qIdx) : '';
  }},
}};
{core_script}
getMetricsForState = () => ({json.dumps(_baseline_safe_metrics())});
function resetDefaults() {{
  Object.assign(state, {{
    view: 'plan',
    s_kiln_mm: 300,
    u_kiln_side: 0.8,
    exhaust_fan_cfm: 400,
    d_hood_inlet_mm: 800,
    z_hood_drop_mm: 400,
    d_reflector_mm: 800,
    g_floor_reflector_mm: 20,
    t_reflector_calsil_mm: 0,
    w_floor_pad_mm: 1200,
    t_floor_pad_mm: 20,
    w_door_open_mm: 150,
  }});
}}
if (typeof loadStateFromUrl !== 'function') {{
  throw new Error('loadStateFromUrl is not defined');
}}
if (typeof syncUrlFromState !== 'function') {{
  throw new Error('syncUrlFromState is not defined');
}}

resetDefaults();
loadStateFromUrl('?view=front&skiln=200&ukiln=0.3&fan=450&hood=600&hdrop=150&dref=800&gref=20&tref=10&tpad=0&door=0');
const fullParsed = {{ ...state }};

resetDefaults();
loadStateFromUrl('?dref=0');
const drefZero = {{ ...state }};

resetDefaults();
loadStateFromUrl('?dref=800&gref=0&tref=10');
const trefNeedsGap = {{ ...state }};

resetDefaults();
loadStateFromUrl('?view=bogus&skiln=999&ukiln=abc&fan=-5');
const invalidIgnored = {{ ...state }};

resetDefaults();
setupControls();
syncControlButtons();
__buttonMap['ctrl-view'].find(b => b.dataset.val === 'side').click();
__buttonMap['ctrl-skiln'].find(b => b.dataset.val === '200').click();
__buttonMap['ctrl-dref'].find(b => b.dataset.val === '0').click();
__buttonMap['ctrl-hdrop'].find(b => b.dataset.val === '150').click();
const clickedState = {{ ...state }};
const clickedUrl = __replaceCalls[__replaceCalls.length - 1] || '';

resetDefaults();
const qIdx = clickedUrl.indexOf('?');
loadStateFromUrl(qIdx >= 0 ? clickedUrl.slice(qIdx) : '');
const roundTripState = {{ ...state }};

process.stdout.write(JSON.stringify({{
  fullParsed,
  drefZero,
  trefNeedsGap,
  invalidIgnored,
  replaceCount: __replaceCalls.length,
  clickedUrl,
  clickedState,
  roundTripState,
}}));
"""
    proc = subprocess.run(
        ["/usr/bin/node", "-e", harness],
        capture_output=True,
        text=True,
    )
    self.assertEqual(
        proc.returncode,
        0,
        f"Node harness failed (rc={proc.returncode}): {proc.stderr}",
    )
    res = json.loads(proc.stdout)

    self.assertEqual(res["fullParsed"]["view"], "front")
    self.assertEqual(res["fullParsed"]["s_kiln_mm"], 200)
    self.assertAlmostEqual(res["fullParsed"]["u_kiln_side"], 0.3)
    self.assertEqual(res["fullParsed"]["exhaust_fan_cfm"], 450)
    self.assertEqual(res["fullParsed"]["d_hood_inlet_mm"], 600)
    self.assertEqual(res["fullParsed"]["z_hood_drop_mm"], 150)
    self.assertEqual(res["fullParsed"]["d_reflector_mm"], 800)
    self.assertEqual(res["fullParsed"]["g_floor_reflector_mm"], 20)
    self.assertEqual(res["fullParsed"]["t_reflector_calsil_mm"], 10)
    self.assertEqual(res["fullParsed"]["t_floor_pad_mm"], 0)
    self.assertEqual(res["fullParsed"]["w_door_open_mm"], 0)

    self.assertEqual(res["drefZero"]["d_reflector_mm"], 0)
    self.assertEqual(res["drefZero"]["g_floor_reflector_mm"], 0)
    self.assertEqual(res["drefZero"]["t_reflector_calsil_mm"], 0)

    self.assertEqual(res["trefNeedsGap"]["d_reflector_mm"], 800)
    self.assertEqual(res["trefNeedsGap"]["t_reflector_calsil_mm"], 10)
    self.assertEqual(res["trefNeedsGap"]["g_floor_reflector_mm"], 20)

    self.assertEqual(res["invalidIgnored"]["view"], "plan")
    self.assertEqual(res["invalidIgnored"]["s_kiln_mm"], 300)
    self.assertAlmostEqual(res["invalidIgnored"]["u_kiln_side"], 0.8)
    self.assertEqual(res["invalidIgnored"]["exhaust_fan_cfm"], 400)

    self.assertGreaterEqual(res["replaceCount"], 4)
    self.assertIn("view=side", res["clickedUrl"])
    self.assertIn("skiln=200", res["clickedUrl"])
    self.assertIn("dref=0", res["clickedUrl"])
    self.assertIn("hdrop=150", res["clickedUrl"])
    self.assertEqual(res["roundTripState"], res["clickedState"])


if __name__ == "__main__":
  unittest.main()

