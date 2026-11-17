"""Unit tests verifying safety threshold consistency in kiln_explorer.html."""

from html import parser as html_parser
import json
import os
import pathlib
import re
import subprocess
import unittest
from urllib import parse as urllib_parse

import run_sweep
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


class _ButtonGroupParser(html_parser.HTMLParser):
  """Extracts <button data-val="..."> values inside .btn-group controls."""

  def __init__(self) -> None:
    super().__init__()
    self.groups: dict[str, list[str]] = {}
    self._current_group: str | None = None
    self._div_depth = 0

  def handle_starttag(
      self, tag: str, attrs: list[tuple[str, str | None]]
  ) -> None:
    attr_map = dict(attrs)
    if tag == "div":
      classes = (attr_map.get("class") or "").split()
      div_id = attr_map.get("id")
      if "btn-group" in classes and div_id and div_id.startswith("ctrl-"):
        self._current_group = div_id
        self._div_depth = 1
        self.groups[div_id] = []
      elif self._current_group is not None:
        self._div_depth += 1
    elif tag == "button" and self._current_group is not None:
      val = attr_map.get("data-val")
      if val is not None:
        self.groups[self._current_group].append(val)

  def handle_endtag(self, tag: str) -> None:
    if tag == "div" and self._current_group is not None:
      self._div_depth -= 1
      if self._div_depth <= 0:
        self._current_group = None


def _parse_html_button_groups(html_text: str) -> dict[str, list[str]]:
  parser = _ButtonGroupParser()
  parser.feed(html_text)
  parser.close()
  return parser.groups


def _default_button_groups() -> dict[str, list[str]]:
  return {
      "ctrl-view": ["side", "front", "plan"],
      "ctrl-skiln": ["200", "300"],
      "ctrl-ukiln": ["0.8", "0.5", "0.3"],
      "ctrl-fan": ["300", "350", "400"],
      "ctrl-hood": ["600", "700", "800"],
      "ctrl-hdrop": ["150", "400", "600"],
      "ctrl-dref": ["0", "800"],
      "ctrl-gref": ["0", "20"],
      "ctrl-tref": ["0", "10"],
      "ctrl-tpad": ["0", "20"],
      "ctrl-door": ["0", "150"],
  }


def _run_dom_explorer_js(
    button_groups: dict[str, list[str]], body_js: str
) -> dict[str, object]:
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
const __buttonSpec = {json.dumps(button_groups)};
const __buttonMap = {{}};
for (const [gid, vals] of Object.entries(__buttonSpec)) {{
  __buttonMap[gid] = vals.map(v => {{
    const btn = {{
      dataset: {{ val: v }},
      disabled: false,
      _active: false,
      _handlers: {{}},
      classList: {{ toggle(cls, on) {{ if (cls === 'active') btn._active = Boolean(on); }} }},
      addEventListener(ev, fn) {{ btn._handlers[ev] = fn; }},
      click() {{ if (btn._handlers.click) btn._handlers.click(); }},
    }};
    return btn;
  }});
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
    w_floor_pad_mm: 1200,
    t_floor_pad_mm: 20,
    w_door_open_mm: 150,
    exhaust_fan_cfm: 400,
    d_reflector_mm: 800,
    t_reflector_calsil_mm: 0,
    g_floor_reflector_mm: 20,
    d_chimney_mm: 0,
    g_chimney_mm: 0,
    w_shield_mm: 1200,
    d_hood_inlet_mm: 800,
    z_hood_drop_mm: 400,
    bung_captured: true,
  }});
}}
if (typeof loadStateFromUrl !== 'function') {{
  throw new Error('loadStateFromUrl is not defined');
}}
if (typeof syncUrlFromState !== 'function') {{
  throw new Error('syncUrlFromState is not defined');
}}
{body_js}
"""
  proc = subprocess.run(
      ["/usr/bin/node", "-e", harness],
      capture_output=True,
      text=True,
      check=False,
  )
  if proc.returncode != 0:
    raise AssertionError(
        f"Node harness failed (rc={proc.returncode}): {proc.stderr}"
    )
  return json.loads(proc.stdout)


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
    res = _run_dom_explorer_js(
        _default_button_groups(),
        """
resetDefaults();
loadStateFromUrl('?view=front&skiln=200&ukiln=0.3&fan=450&hood=600&hdrop=150&dref=800&gref=20&tref=10&tpad=0&door=0');
const fullParsed = { ...state };

resetDefaults();
loadStateFromUrl('?dref=0');
const drefZero = { ...state };

resetDefaults();
loadStateFromUrl('?dref=800&gref=0&tref=10');
const trefNeedsGap = { ...state };

resetDefaults();
loadStateFromUrl('?view=bogus&skiln=999&ukiln=abc&fan=-5');
const invalidIgnored = { ...state };

resetDefaults();
setupControls();
syncControlButtons();
__buttonMap['ctrl-view'].find(b => b.dataset.val === 'side').click();
__buttonMap['ctrl-skiln'].find(b => b.dataset.val === '200').click();
__buttonMap['ctrl-dref'].find(b => b.dataset.val === '0').click();
__buttonMap['ctrl-hdrop'].find(b => b.dataset.val === '150').click();
const clickedState = { ...state };
const clickedUrl = __replaceCalls[__replaceCalls.length - 1] || '';

resetDefaults();
const qIdx = clickedUrl.indexOf('?');
loadStateFromUrl(qIdx >= 0 ? clickedUrl.slice(qIdx) : '');
const roundTripState = { ...state };

process.stdout.write(JSON.stringify({
  fullParsed,
  drefZero,
  trefNeedsGap,
  invalidIgnored,
  replaceCount: __replaceCalls.length,
  clickedUrl,
  clickedState,
  roundTripState,
}));
""",
    )

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

  def test_all_dataset_params_in_url_and_unmatched_controls_unselected(
      self,
  ) -> None:
    html_text = _find_kiln_explorer_path().read_text(encoding="utf-8")
    self.assertEqual(
        _parse_html_button_groups(html_text), _default_button_groups()
    )

    omitted_hood_buttons = _default_button_groups()
    omitted_hood_buttons["ctrl-hood"] = ["700", "800"]

    res = _run_dom_explorer_js(
        omitted_hood_buttons,
        """
const datasetProps = URL_PARAM_SPECS
  .filter(s => s.key !== 'view')
  .map(s => s.prop);
resetDefaults();
const defaultQuery = buildStateQueryString();

resetDefaults();
loadStateFromUrl('?pad=1200&dchim=0&gchim=0&wshield=1200&bung=true');
const shortSingleState = { ...state };

resetDefaults();
loadStateFromUrl(
  '?s_kiln_mm=200.0&u_kiln_side=0.5&w_floor_pad_mm=1200.0&t_floor_pad_mm=0.0' +
  '&w_door_open_mm=0.0&exhaust_fan_cfm=350.0&d_reflector_mm=800.0' +
  '&t_reflector_calsil_mm=0.0&g_floor_reflector_mm=0.0&d_chimney_mm=0.0' +
  '&g_chimney_mm=0.0&w_shield_mm=1200.0&d_hood_inlet_mm=700.0' +
  '&z_hood_drop_mm=600.0&bung_captured=True'
);
const fullCsvState = { ...state };

// Case A: dataset-valid hood=600 when #ctrl-hood omits "600".
setupControls();
resetDefaults();
loadStateFromUrl('?hood=600');
syncControlButtons();
syncUrlFromState();
const caseA = {
  hoodVal: state.d_hood_inlet_mm,
  hoodLabel: __elements['lbl-hood'].textContent,
  hoodActive: __buttonMap['ctrl-hood'].map(b => b._active),
  urlBeforeClick: __replaceCalls[__replaceCalls.length - 1] || '',
};
__buttonMap['ctrl-door'].find(b => b.dataset.val === '0').click();
caseA.urlAfterDoorClick = __replaceCalls[__replaceCalls.length - 1] || '';

// Check real dataset fan values 300 (matched button) and 450 (unmatched button)
// before mutating SWEEP_DIMS.
resetDefaults();
loadStateFromUrl('?fan=300');
syncControlButtons();
syncUrlFromState();
const realFan300 = {
  fanVal: state.exhaust_fan_cfm,
  fanLabel: __elements['lbl-fan'].textContent,
  fanActive: __buttonMap['ctrl-fan'].map(b => b._active),
  url: __replaceCalls[__replaceCalls.length - 1] || '',
};

resetDefaults();
loadStateFromUrl('?fan=450');
syncControlButtons();
syncUrlFromState();
const realFan450 = {
  fanVal: state.exhaust_fan_cfm,
  fanLabel: __elements['lbl-fan'].textContent,
  fanActive: __buttonMap['ctrl-fan'].map(b => b._active),
  url: __replaceCalls[__replaceCalls.length - 1] || '',
};

// Case B: dynamically extend SWEEP_DIMS with values not in HTML buttons.
SWEEP_DIMS.fan[500] = 0;
SWEEP_DIMS.wshield[1000] = 0;
SWEEP_DIMS.bung[false] = 0;
SWEEP_DIMS.refCombo['600|5|15'] = 4;

resetDefaults();
loadStateFromUrl('?fan=500&wshield=1000&bung=false&dref=600&tref=5&gref=15');
syncControlButtons();
syncUrlFromState();
const caseB = {
  stateAfterValid: { ...state },
  labels: {
    fan: __elements['lbl-fan'].textContent,
    dref: __elements['lbl-dref'].textContent,
    tref: __elements['lbl-tref'].textContent,
    gref: __elements['lbl-gref'].textContent,
  },
  activeFlags: {
    fan: __buttonMap['ctrl-fan'].map(b => b._active),
    dref: __buttonMap['ctrl-dref'].map(b => b._active),
    tref: __buttonMap['ctrl-tref'].map(b => b._active),
    gref: __buttonMap['ctrl-gref'].map(b => b._active),
  },
  urlAfterValid: __replaceCalls[__replaceCalls.length - 1] || '',
};
loadStateFromUrl('?fan=999&wshield=777&dchim=25');
caseB.stateAfterInvalid = { ...state };

// Case C: clicking a button afterward re-selects that control cleanly.
__buttonMap['ctrl-fan'].find(b => b.dataset.val === '400').click();
const caseC = {
  fanVal: state.exhaust_fan_cfm,
  fanActive: __buttonMap['ctrl-fan'].map(b => b._active),
  urlAfterClick: __replaceCalls[__replaceCalls.length - 1] || '',
};

process.stdout.write(JSON.stringify({
  datasetProps,
  defaultQuery,
  shortSingleState,
  fullCsvState,
  caseA,
  realFan300,
  realFan450,
  caseB,
  caseC,
}));
""",
    )

    self.assertEqual(
        set(res["datasetProps"]),
        set(run_sweep.SWEEP_PARAM_COLUMNS),
    )

    default_query_params = urllib_parse.parse_qs(
        res["defaultQuery"].lstrip("?"), keep_blank_values=True
    )
    expected_url_keys = {
        "view",
        "skiln",
        "ukiln",
        "pad",
        "tpad",
        "door",
        "fan",
        "dref",
        "tref",
        "gref",
        "dchim",
        "gchim",
        "wshield",
        "hood",
        "hdrop",
        "bung",
    }
    self.assertEqual(set(default_query_params.keys()), expected_url_keys)

    self.assertEqual(res["shortSingleState"]["w_floor_pad_mm"], 1200)
    self.assertEqual(res["shortSingleState"]["d_chimney_mm"], 0)
    self.assertEqual(res["shortSingleState"]["g_chimney_mm"], 0)
    self.assertEqual(res["shortSingleState"]["w_shield_mm"], 1200)
    self.assertIs(res["shortSingleState"]["bung_captured"], True)

    csv_state = res["fullCsvState"]
    self.assertEqual(csv_state["s_kiln_mm"], 200)
    self.assertAlmostEqual(csv_state["u_kiln_side"], 0.5)
    self.assertEqual(csv_state["w_floor_pad_mm"], 1200)
    self.assertEqual(csv_state["t_floor_pad_mm"], 0)
    self.assertEqual(csv_state["w_door_open_mm"], 0)
    self.assertEqual(csv_state["exhaust_fan_cfm"], 350)
    self.assertEqual(csv_state["d_reflector_mm"], 800)
    self.assertEqual(csv_state["t_reflector_calsil_mm"], 0)
    self.assertEqual(csv_state["g_floor_reflector_mm"], 0)
    self.assertEqual(csv_state["d_chimney_mm"], 0)
    self.assertEqual(csv_state["g_chimney_mm"], 0)
    self.assertEqual(csv_state["w_shield_mm"], 1200)
    self.assertEqual(csv_state["d_hood_inlet_mm"], 700)
    self.assertEqual(csv_state["z_hood_drop_mm"], 600)
    self.assertIs(csv_state["bung_captured"], True)

    # Case A assertions.
    self.assertEqual(res["caseA"]["hoodVal"], 600)
    self.assertEqual(res["caseA"]["hoodLabel"], "600 mm")
    self.assertEqual(res["caseA"]["hoodActive"], [False, False])
    self.assertIn("hood=600", res["caseA"]["urlBeforeClick"])
    self.assertIn("hood=600", res["caseA"]["urlAfterDoorClick"])
    self.assertIn("door=0", res["caseA"]["urlAfterDoorClick"])

    # Real 300 / 450 CFM dataset values against 300/350/400 UI buttons.
    self.assertEqual(res["realFan300"]["fanVal"], 300)
    self.assertEqual(res["realFan300"]["fanLabel"], "300 CFM")
    self.assertEqual(res["realFan300"]["fanActive"], [True, False, False])
    self.assertIn("fan=300", res["realFan300"]["url"])

    self.assertEqual(res["realFan450"]["fanVal"], 450)
    self.assertEqual(res["realFan450"]["fanLabel"], "450 CFM")
    self.assertEqual(res["realFan450"]["fanActive"], [False, False, False])
    self.assertIn("fan=450", res["realFan450"]["url"])

    # Case B assertions.
    sb = res["caseB"]["stateAfterValid"]
    self.assertEqual(sb["exhaust_fan_cfm"], 500)
    self.assertEqual(sb["w_shield_mm"], 1000)
    self.assertIs(sb["bung_captured"], False)
    self.assertEqual(sb["d_reflector_mm"], 600)
    self.assertEqual(sb["t_reflector_calsil_mm"], 5)
    self.assertEqual(sb["g_floor_reflector_mm"], 15)
    self.assertEqual(res["caseB"]["labels"]["fan"], "500 CFM")
    self.assertEqual(res["caseB"]["labels"]["dref"], "600 mm")
    self.assertEqual(res["caseB"]["labels"]["tref"], "5 mm")
    self.assertEqual(res["caseB"]["labels"]["gref"], "15 mm")
    for ctrl in ("fan", "dref", "tref", "gref"):
      self.assertEqual(res["caseB"]["activeFlags"][ctrl], [False] * len(res["caseB"]["activeFlags"][ctrl]))
    self.assertIn("fan=500", res["caseB"]["urlAfterValid"])
    self.assertIn("wshield=1000", res["caseB"]["urlAfterValid"])
    self.assertIn("bung=false", res["caseB"]["urlAfterValid"])

    sb_invalid = res["caseB"]["stateAfterInvalid"]
    self.assertEqual(sb_invalid["exhaust_fan_cfm"], 500)
    self.assertEqual(sb_invalid["w_shield_mm"], 1000)
    self.assertEqual(sb_invalid["d_chimney_mm"], 0)

    # Case C assertions.
    self.assertEqual(res["caseC"]["fanVal"], 400)
    self.assertEqual(res["caseC"]["fanActive"], [False, False, True])
    self.assertIn("fan=400", res["caseC"]["urlAfterClick"])


if __name__ == "__main__":
  unittest.main()

