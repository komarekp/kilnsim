# Kiln Corner Thermal & Airflow Modelling Design Document

## 1. Purpose & Scope

This document specifies the physical, geometric, and mathematical model for simulating steady-state coupled heat transfer and airflow around a **Rohde Ecotop 60** pottery kiln installed in the back-left corner of a garden studio outbuilding.

The goal of the model is to allow rapid parametric iteration and safety verification over:

1. **Wall heat shield dimensions and standoffs:** Shield width, floor and ceiling gaps, side sealing, and behind-shield chimney depth (supporting both open buoyant natural-draft channels and flush-mounted zero-gap configurations).
2. **Floor protection configurations:** Square non-combustible topping pad dimensions (`w_floor_pad`, `t_floor_pad`), circular reflector diameter (`d_reflector`), reflector backing thickness (`t_reflector_calsil`), and under-reflector air gap (`g_floor_reflector`).
3. **Active exhaust hood parameters:** Canopy diameter (`d_hood_inlet`), throat duct diameter (`d_hood_duct`), mounting height drop (`z_hood_drop`), hood centerline position (`x_hood`, `y_hood`), and blower flow rate in CFM (evaluating plume capture efficiency from 0 to 800 CFM).
4. **Side exhaust dilution bung:** Dilution cup geometry and capture state (`bung_captured` flag, 0 W room load when captured vs sensible heat release when open).
5. **SSR electrical enclosure:** Louvered housing orientation, casing and internal air thermal protection.
6. **Two-zone stratified room ventilation:** Buoyancy exchange through an ajar sliding door, convective ceiling stratification, and envelope conduction.

---

## 2. Coordinate System & Architectural Layout

### 2.1 Global 3D Right-Handed Cartesian Coordinate Frame

The origin $(0, 0, 0)$ is defined at the **inner structural corner of the bare AAC block walls at the top surface of the continuous floor screed**, forming a standard right-handed Cartesian coordinate system:

- **+x axis (Left Wall, plane y = 0):** Runs horizontally along the left boundary wall toward the front sliding glass doors ($0 \le x \le L\_{\text{room}}$).
- **+y axis (Back Wall, plane x = 0):** Runs horizontally along the back wall toward the internal partition wall on the right ($0 \le y \le W\_{\text{room}}$).
- **+z axis (Vertical):** Runs vertically upward from the top of the structural floor screed ($z = 0$) to the studio ceiling ($z = H\_{\text{wall}}$).

### 2.2 Wall Zones (No Gypsum Above or Below Shield)

Each of the two perpendicular corner walls is divided horizontally into two distinct structural zones:

1. **Corner Shield Column ($0 \le x \le W\_{\text{shield,left}}$ on Left Wall; $0 \le y \le W\_{\text{shield,back}}$ on Back Wall):**
   - The structural wall is **bare 215 mm Airtec AAC block** from the screed datum ($z = 0$) to the ceiling ($z = H\_{\text{wall}}$). No gypsum plasterboard is installed above, below, or behind the shield in this column.
   - Mounted at a standoff distance $d\_{\text{chimney}}$ in front of the bare AAC block is a calcium silicate board (Promatect-H) of thickness $t\_{\text{shield}}$.
   - The shield starts at elevation $z\_{\text{shield,bot}} = g\_{\text{floor}}$ above the screed datum and ends at $z\_{\text{shield,top}} = H\_{\text{wall}} - g\_{\text{ceil}}$, giving a derived shield height:
     $$H\_{\text{shield}} = H\_{\text{wall}} - g\_{\text{floor}} - g\_{\text{ceil}}$$
   - The inner front faces of the two shields define the planes:
     - Left Wall shield: $y\_{\text{shield}} = d\_{\text{chimney}} + t\_{\text{shield}}$
     - Back Wall shield: $x\_{\text{shield}} = d\_{\text{chimney}} + t\_{\text{shield}}$
   - Along the horizontal axes, the left shield starts at $x = d\_{\text{chimney}} + t\_{\text{shield}}$ (abutting the inner front face of the back shield) and spans to $x = d\_{\text{chimney}} + t\_{\text{shield}} + W\_{\text{shield,left}}$. The back shield starts at $y = d\_{\text{chimney}}$ and spans to $y = d\_{\text{chimney}} + W\_{\text{shield,back}}$.
2. **Exposed Gypsum Zone ($x > x\_{\text{shield,left,end}}$ on Left Wall; $y > y\_{\text{shield,back,end}}$ on Back Wall):**
   - Beyond the horizontal width of the shields, standard Class A2 gypsum plasterboard of thickness $t\_{\text{gypsum}}$ is bonded directly to the inner face of the AAC block from floor to ceiling.

### 2.3 Floor & Subfloor Zones (Vertical Datum Audit)

The studio subfloor below $z = 0$ consists of a 150 mm structural concrete slab, a continuous layer of Vacuum Insulation Panels (VIPs) of thickness $t\_{\text{vip}}$ and thermal conductivity $k\_{\text{vip}}$, and a fiber-reinforced screed of thickness $t\_{\text{screed}}$ ($k\_{\text{screed}} \approx 1.2\ \text{W/m}\cdot\text{K}$) whose top surface defines $z = 0$:

1. **Corner Floor Pad Zone ($0 \le x \le W\_{\text{floor\_pad}}, \, 0 \le y \le W\_{\text{floor\_pad}}$):**
   - Industrial vinyl flooring is omitted in this square corner zone.
   - Over the screed ($z = 0$) sits a non-combustible topping layer (porcelain tile, quarry tile, or concrete topping pad) of thickness $t\_{\text{floor\_pad}} \ge 0\text{ mm}$, whose top surface defines the elevation:
     $$z\_{\text{pad,top}} = t\_{\text{floor\_pad}}$$
   - Both the kiln stand legs and the circular floor reflector assembly rest independently atop this pad surface ($z = z\_{\text{pad,top}}$).
2. **Circular Floor Reflector ($r \le D\_{\text{reflector}}/2$ centered on the kiln axis):**
   - Sits between the kiln stand legs directly beneath the kiln drum. Consists of a polished stainless steel top sheet ($\varepsilon\_{\text{reflector}} = 0.15, \rho\_s = 0.85$) bonded to a calcium silicate backing of thickness $t\_{\text{reflector\_calsil}} \ge 0\text{ mm}$.
   - Rests on non-combustible spacer feet of height $g\_{\text{floor\_reflector}} \ge 0\text{ mm}$ atop the floor pad ($z = z\_{\text{pad,top}}$).
   - Underside of reflector:
     $$z\_{\text{reflector,bot}} = z\_{\text{pad,top}} + g\_{\text{floor\_reflector}} = t\_{\text{floor\_pad}} + g\_{\text{floor\_reflector}}$$
   - Top reflective surface of reflector:
     $$z\_{\text{reflector,top}} = z\_{\text{reflector,bot}} + t\_{\text{reflector\_calsil}} = t\_{\text{floor\_pad}} + g\_{\text{floor\_reflector}} + t\_{\text{reflector\_calsil}}$$
3. **Kiln Drum Base & Lid Elevations:**
   - The underside of the kiln drum bottom plate sits at a clear height $h\_{\text{kiln\_base}}$ above the top surface of the floor pad ($z\_{\text{pad,top}}$), representing the physical stand height:
     $$z\_{\text{drum,bot}} = z\_{\text{pad,top}} + h\_{\text{kiln\_base}} = t\_{\text{floor\_pad}} + h\_{\text{kiln\_base}}$$
   - Total closed kiln cylinder height: $H\_{\text{kiln}} = H\_{\text{drum}} + t\_{\text{lid}}$.
   - Top flat surface of the closed kiln lid sits at elevation:
     $$z\_{\text{kiln,top}} = z\_{\text{drum,bot}} + H\_{\text{drum}} + t\_{\text{lid}}$$
4. **Surrounding Vinyl Floor ($x > W\_{\text{floor\_pad}}$ or $y > W\_{\text{floor\_pad}}$):**
   - Standard industrial vinyl flooring rests directly at $z = 0$ bonded to the screed outside the square pad boundary.

### 2.4 Kiln, SSR Box, and Bung Orientation

- **Kiln Cylinder:** Outer diameter $D\_{\text{kiln}}$, radius $R\_{\text{kiln}} = D\_{\text{kiln}}/2$. Vertical centerline axis located at:
  $$x\_k = x\_{\text{shield,front}} + s\_{\text{kiln,back}} + R\_{\text{kiln}} = d\_{\text{chimney}} + t\_{\text{shield}} + s\_{\text{kiln,back}} + \frac{D\_{\text{kiln}}}{2}$$
  $$y\_k = y\_{\text{shield,front}} + s\_{\text{kiln,left}} + R\_{\text{kiln}} = d\_{\text{chimney}} + t\_{\text{shield}} + s\_{\text{kiln,left}} + \frac{D\_{\text{kiln}}}{2}$$
- **Right-Handed Polar Angle Convention ($\theta = \text{atan2}(y - y\_k, \, x - x\_k)$):**
  - $\theta = 0^\circ$: Points in $+x$ direction (parallel to left wall, straight toward front sliding glass doors).
  - $\theta = 90^\circ$: Points in $+y$ direction (parallel to back wall, out into studio).
  - $\theta = 180^\circ$: Points into back wall ($-x$ direction).
  - $\theta = -90^\circ$ ($270^\circ$): Points into left wall ($-y$ direction).
- **SSR Enclosure:** Positioned at $\theta\_{\text{SSR}} = -45^\circ$ (front-left quadrant) with vertical offset $z\_{\text{SSR\_offset}}$ from the kiln drum bottom plate to the bottom face of the SSR box:
  $$z\_{\text{SSR,bot}} = z\_{\text{drum,bot}} + z\_{\text{SSR\_offset}}$$
  $$z\_{\text{SSR,top}} = z\_{\text{SSR,bot}} + H\_{\text{SSR}}$$
  Outer rear face has tangential width $W\_{\text{SSR}}$ and flat side-panel flanges of uniform depth $D\_{\text{SSR}}$.
- **Exhaust Bung:** Positioned at $\theta\_{\text{bung}} = 0^\circ$ (pointing in $+x$ direction toward front doors) at radial center $r\_{\text{bung}}$ and elevation:
  $$z\_{\text{bung}} = z\_{\text{drum,bot}} + z\_{\text{bung\_offset}}$$

---

## 3. Complete Parameter Specification & Validity Ranges

### 3.1 Wall, Shield & Chimney Parameters

| Parameter (`config_field`) | Baseline Default | Valid Range | Units | Physical Role & Model Validity Limits |
|:--------------------------|:-----------------|:------------|:------|:---------------------------------------------------------------------------------------------------|
| $W\_{\text{shield,left}}$ \newline `w_shield_left_mm` | 800.0 | $\ge 200.0$ | mm | Width of Calsil shield along left wall (+x axis) from corner. |
| $W\_{\text{shield,back}}$ \newline `w_shield_back_mm` | 800.0 | $\ge 200.0$ | mm | Width of Calsil shield along back wall (+y axis) from corner. |
| $g\_{\text{floor}}$ \newline `g_floor_mm` | 50.0 | $\ge 0.0$ | mm | Inlet air gap between screed ($z = 0$) and bottom edge of shield. 0 mm simulates a floor-sealed cavity. |
| $g\_{\text{ceil}}$ \newline `g_ceil_mm` | 50.0 | $\ge 0.0$ | mm | Outlet air gap between top edge of shield and ceiling ($z = H\_{\text{wall}}$). 0 mm simulates a ceiling-sealed cavity. |
| $d\_{\text{chimney}}$ \newline `d_chimney_mm` | 25.0 | 0.0 to 100.0 | mm | Air gap between bare AAC block and shield back face. 0 mm simulates flush solid mounting with direct conduction. |
| $t\_{\text{shield}}$ \newline `t_shield_mm` | 20.0 | $\ge 10.0$ | mm | Thickness of Promatect-H calcium silicate board. |
| $f\_{\text{block}}$ \newline `chimney_blockage_fraction` | 0.08 | 0.0 to 1.0 | -- | Fraction of chimney cross-section obstructed by vertical mounting battens. |
| $\text{seal}\_{\text{sides}}$ \newline `seal_chimney_sides` | `true` | `true` / `false` | bool | If `true`, vertical outer edges are sealed; if `false`, lateral room air entrains through side slot. |
| $H\_{\text{wall}}$ \newline `h_wall_mm` | 2345.0 | $\ge 1800.0$ | mm | Total interior wall height from structural screed ($z = 0$) to ceiling. |
| $t\_{\text{aac}}$ \newline `t_aac_mm` | 215.0 | $\ge 100.0$ | mm | Thickness of Airtec aerated concrete block wall. |
| $t\_{\text{gypsum}}$ \newline `t_gypsum_mm` | 12.5 | $\ge 9.5$ | mm | Thickness of Class A2 gypsum plasterboard outside shield zone. |

### 3.2 Kiln, Bung & SSR Parameters

| Parameter (`config_field`) | Baseline Default | Valid Range | Units | Physical Role & Model Validity Limits |
|:--------------------------|:-----------------|:------------|:------|:---------------------------------------------------------------------------------------------------|
| $s\_{\text{kiln,left}}$ \newline `s_kiln_left_mm` | 205.0 | $\ge 100.0$ | mm | Clear horizontal standoff from left shield front face to kiln jacket. |
| $s\_{\text{kiln,back}}$ \newline `s_kiln_back_mm` | 205.0 | $\ge 100.0$ | mm | Clear horizontal standoff from back shield front face to kiln jacket. |
| $D\_{\text{kiln}}$ \newline `d_kiln_mm` | 700.0 | 400.0 to 1200.0 | mm | Outer diameter of stainless steel kiln cylinder jacket. |
| $H\_{\text{drum}}$ \newline `h_drum_mm` | 580.0 | 300.0 to 1000.0 | mm | Height of main kiln cylinder body excluding lid. |
| $t\_{\text{lid}}$ \newline `t_lid_mm` | 105.0 | 50.0 to 200.0 | mm | Thickness of closed refractory kiln lid. |
| $h\_{\text{kiln\_base}}$ \newline `h_kiln_base_mm` | 210.0 | $\ge 50.0$ | mm | Clear height from floor pad top ($z\_{\text{pad,top}}$) to kiln drum bottom plate. |
| $T\_{\text{int}}$ \newline `t_internal_c` | 1280.0 | 800.0 to 1350.0 | deg C | Steady-state internal firing temperature (Cone 10 firing). |
| $U\_{\text{kiln,side}}$ \newline `u_kiln_side` | 0.80 | 0.40 to 2.50 | $\text{W/m}^2\text{K}$ | Effective thermal transmittance of composite kiln wall ($0.80\ \text{W/m}^2\text{K}$ gives $\sim 155^\circ\text{C}$ skin). |
| $U\_{\text{kiln,top}}$ \newline `u_kiln_top` | 2.00 | 0.80 to 4.00 | $\text{W/m}^2\text{K}$ | Effective thermal transmittance of unbacked refractory brick lid. |
| $U\_{\text{kiln,bot}}$ \newline `u_kiln_bot` | 2.00 | 0.80 to 4.00 | $\text{W/m}^2\text{K}$ | Effective thermal transmittance of kiln bottom floor slab. |
| $\theta\_{\text{bung}}$ \newline `bung_angle_deg` | 0.0 | -180.0 to 180.0 | deg | Azimuthal angle of exhaust bung ($0^\circ$ faces sliding doors along +x). |
| $r\_{\text{bung}}$ \newline `r_bung_mm` | 415.0 | $\ge D\_{\text{kiln}}/2$ | mm | Radial distance from kiln axis to dilution cup center. |
| $z\_{\text{bung\_offset}}$ \newline `z_bung_offset_mm` | 540.0 | 0.0 to $H\_{\text{drum}}$ | mm | Vertical height from kiln bottom plate to open top rim of dilution cup. |
| $\text{bung}\_{\text{cap}}$ \newline `bung_captured` | `false` | `true` / `false` | bool | If `true`, bung exhaust is captured directly into duct ($0\text{ W}$ room load). |
| $\dot{Q}\_{\text{bung}}$ \newline `q_bung_open_w` | 110.0 | 0.0 to 500.0 | W | Sensible convective heat release rate exiting open exhaust bung. |
| $\theta\_{\text{SSR}}$ \newline `ssr_angle_deg` | -45.0 | -180.0 to 180.0 | deg | Azimuthal mounting angle of SSR box ($-45^\circ$ faces front-left quadrant). |
| $z\_{\text{SSR\_offset}}$ \newline `z_ssr_offset_mm` | 100.0 | 0.0 to $H\_{\text{drum}}$ | mm | Vertical distance from kiln bottom plate to bottom face of SSR enclosure. |
| $W\_{\text{SSR}}$ \newline `w_ssr_mm` | 190.0 | 100.0 to 350.0 | mm | Tangential width across outer back face of SSR enclosure. |
| $D\_{\text{SSR}}$ \newline `d_ssr_mm` | 146.5 | 50.0 to 250.0 | mm | Depth of flat side flanges from kiln jacket to outer back face. |
| $H\_{\text{SSR}}$ \newline `h_ssr_mm` | 430.0 | 200.0 to 600.0 | mm | Vertical height of louvered SSR enclosure. |
| $\dot{Q}\_{\text{SSR,int}}$ \newline `q_ssr_internal_w` | 18.8 | 0.0 to 50.0 | W | Internal relay Joule heating dissipation ($15.7\text{ A} \times 1.2\text{ V}$). |

### 3.3 Floor, Subfloor VIP & Reflector Parameters

| Parameter (`config_field`) | Baseline Default | Valid Range | Units | Physical Role & Model Validity Limits |
|:--------------------------|:-----------------|:------------|:------|:---------------------------------------------------------------------------------------------------|
| $D\_{\text{reflector}}$ \newline `d_reflector_mm` | 730.0 | 500.0 to 1200.0 | mm | Outer diameter of circular polished stainless steel + CalSil floor reflector. |
| $t\_{\text{reflector}}$ \newline `t_reflector_calsil_mm` | 20.0 | $\ge 0.0$ | mm | Thickness of CalSil backing inside circular reflector (0 mm for bare sheet). |
| $g\_{\text{floor\_reflector}}$ \newline `g_floor_reflector_mm` | 15.0 | $\ge 0.0$ | mm | Air gap beneath reflector pan formed by spacer feet (0 mm for flush contact). |
| $W\_{\text{floor\_pad}}$ \newline `w_floor_pad_mm` | 1000.0 | $\ge x\_k + R\_{\text{kiln}}$ | mm | Side width of square non-vinyl topping pad in corner ($0 \le x, y \le W\_{\text{floor\_pad}}$). |
| $t\_{\text{floor\_pad}}$ \newline `t_floor_pad_mm` | 20.0 | $\ge 0.0$ | mm | Thickness of concrete or tile topping pad above screed datum $z = 0$. |
| $t\_{\text{screed}}$ \newline `t_screed_mm` | 30.0 | 20.0 to 100.0 | mm | Thickness of fiber-reinforced structural screed layer. |
| $t\_{\text{vip}}$ \newline `t_vip_mm` | 20.0 | 10.0 to 50.0 | mm | Thickness of Vacuum Insulation Panel (VIP) subfloor layer. |
| $k\_{\text{vip}}$ \newline `k_vip` | 0.007 | 0.004 to 0.010 | $\text{W/m}\cdot\text{K}$ | Center-of-panel thermal conductivity of subfloor VIP layer ($R\_{\text{vip}} = t\_{\text{vip}} / k\_{\text{vip}}$). |
| $k\_{\text{pad}}$ \newline `k_pad` | 1.40 | 0.80 to 2.50 | $\text{W/m}\cdot\text{K}$ | Thermal conductivity of concrete or porcelain tile floor pad. |
| $k\_{\text{screed}}$ \newline `k_screed` | 1.20 | 0.80 to 1.80 | $\text{W/m}\cdot\text{K}$ | Thermal conductivity of fiber-reinforced floor screed. |

### 3.4 Hood, Ventilation & Room Envelope Parameters

| Parameter (`config_field`) | Baseline Default | Valid Range | Units | Physical Role & Model Validity Limits |
|:--------------------------|:-----------------|:------------|:------|:---------------------------------------------------------------------------------------------------|
| $\dot{V}\_{\text{fan}}$ \newline `exhaust_fan_cfm` | 350.0 | 0.0 to 1000.0 | CFM | Active volumetric flow rate of overhead exhaust blower ($1\ \text{CFM} = 4.719 \times 10^{-4}\ \text{m}^3/\text{s}$). |
| $D\_{\text{hood}}$ \newline `d_hood_inlet_mm` | 800.0 | 500.0 to 1200.0 | mm | Bottom capture opening diameter of canopy hood. |
| $D\_{\text{duct}}$ \newline `d_hood_duct_mm` | 150.0 | 100.0 to 250.0 | mm | Exhaust duct throat diameter exiting top of hood. |
| $z\_{\text{hood\_drop}}$ \newline `z_hood_drop_mm` | 400.0 | 100.0 to 1000.0 | mm | Distance from ceiling down to hood inlet rim ($z\_{\text{hood,inlet}} = H\_{\text{wall}} - z\_{\text{hood\_drop}}$). |
| $x\_{\text{hood}}, y\_{\text{hood}}$ \newline `x_hood_mm, y_hood_mm` | 600.0, 600.0 | $> 0.0$ | mm | Planar coordinates of vertical hood centerline axis (centered above kiln axis). |
| $L\_{\text{room}}, W\_{\text{room}}$ \newline `l_room_mm, w_room_mm` | 3530.0, 2770.0 | $> 0.0$ | mm | Interior studio dimensions along +x (Left Wall) and +y (Back Wall). |
| $W\_{\text{glass\_door}}$ \newline `w_glass_door_mm` | 2000.0 | $> 0.0$ | mm | Total width of glazed sliding door assembly on front wall ($x = L\_{\text{room}}$). |
| $W\_{\text{door\_gap}}$ \newline `w_door_open_mm` | 150.0 | 0.0 to $W\_{\text{glass\_door}}$ | mm | Open slot width of ajar sliding door during firing. |
| $H\_{\text{door}}$ \newline `h_door_mm` | 2100.0 | $> 0.0$ | mm | Height of sliding glass door opening. |
| $U\_{\text{roof}}$ \newline `u_roof` | 0.18 | 0.10 to 0.50 | $\text{W/m}^2\text{K}$ | Overall thermal transmittance of insulated ceiling/roof structure. |
| $U\_{\text{glass\_door}}$ \newline `u_glass_door` | 1.40 | 1.00 to 3.00 | $\text{W/m}^2\text{K}$ | Thermal transmittance of double-glazed sliding doors. |
| $T\_{\text{ext}}$ \newline `t_ext_c` | 20.0 | -10.0 to 40.0 | deg C | Outdoor / external ambient make-up air temperature. |

### 3.5 Material Emissivities, Conductivities & Discretization

| Parameter (`config_field`) | Baseline Default | Units | Notes |
|:--------------------------|:-----------------|:------|:---------------------------------------------------------------------------------------------------|
| $\varepsilon\_{\text{kiln}}$ \newline `eps_kiln` | 0.60 | -- | Stainless steel outer jacket and lid band ($0.40\text{ new} - 0.70\text{ oxidized}$). |
| $\varepsilon\_{\text{shield}}$ \newline `eps_shield` | 0.90 | -- | Promatect-H calcium silicate board surface. |
| $\varepsilon\_{\text{gypsum}}$ \newline `eps_gypsum` | 0.90 | -- | Class A2 gypsum plasterboard surface. |
| $\varepsilon\_{\text{aac}}$ \newline `eps_aac` | 0.88 | -- | Bare Airtec aerated concrete block surface. |
| $\varepsilon\_{\text{reflector}}$ \newline `eps_reflector_top` | 0.15 | -- | Polished stainless steel top surface ($\rho\_s = 1 - \varepsilon\_{\text{reflector}} = 0.85$). |
| $\varepsilon\_{\text{ssr}}$ \newline `eps_ssr` | 0.60 | -- | Painted sheet-metal SSR electrical enclosure. |
| $k\_{\text{shield}}$ \newline `k_shield` | 0.175 | $\text{W/m}\cdot\text{K}$ | Promatect-H calcium silicate board. |
| $k\_{\text{gypsum}}$ \newline `k_gypsum` | 0.24 | $\text{W/m}\cdot\text{K}$ | Standard gypsum plasterboard. |
| $k\_{\text{aac}}$ \newline `k_aac` | 0.13 | $\text{W/m}\cdot\text{K}$ | Airtec aerated autoclaved concrete block. |
| $\Delta x\_{\text{wall}}, \Delta z\_{\text{wall}}$ \newline `dx_wall_mm, dz_wall_mm` | 50.0, 50.0 | mm | Horizontal and vertical patch grid resolution on Left and Back walls. |
| $N\_\theta, N\_z$ \newline `n_theta_kiln, n_z_kiln` | 24, 6 | -- | Circumferential (15 deg arc) and axial divisions of cylindrical kiln body. |

### 3.6 Derived Parameters (Computed Automatically from User Inputs)

| Derived Parameter (`identifier`) | Mathematical Formula | Units | Physical Meaning |
|:---------------------------------|:-------------------------------------------------------|:------|:-------------------------------------------------------------------|
| $z\_{\text{pad,top}}$ \newline `z_pad_top` | $t\_{\text{floor\_pad}}$ | mm | Elevation of top surface of concrete/tile floor pad above screed ($z = 0$). |
| $z\_{\text{reflector,bot}}$ \newline `z_reflector_bot` | $z\_{\text{pad,top}} + g\_{\text{floor\_reflector}}$ | mm | Elevation of underside of circular floor reflector pan. |
| $z\_{\text{reflector,top}}$ \newline `z_reflector_top` | $z\_{\text{reflector,bot}} + t\_{\text{reflector\_calsil}}$ | mm | Elevation of polished stainless steel top surface of floor reflector. |
| $z\_{\text{drum,bot}}$ \newline `z_drum_bot` | $z\_{\text{pad,top}} + h\_{\text{kiln\_base}}$ | mm | Elevation of underside of kiln drum bottom plate above screed ($z = 0$). |
| $H\_{\text{kiln}}$ \newline `h_kiln_total` | $H\_{\text{drum}} + t\_{\text{lid}}$ | mm | Total closed cylinder height of kiln drum plus refractory lid. |
| $z\_{\text{kiln,top}}$ \newline `z_kiln_top` | $z\_{\text{drum,bot}} + H\_{\text{drum}} + t\_{\text{lid}}$ | mm | Elevation of top surface of closed refractory lid. |
| $z\_{\text{ssr,bot}}$ \newline `z_ssr_bot` | $z\_{\text{drum,bot}} + z\_{\text{SSR\_offset}}$ | mm | Elevation of bottom louver face of SSR electrical enclosure. |
| $z\_{\text{ssr,top}}$ \newline `z_ssr_top` | $z\_{\text{ssr,bot}} + H\_{\text{SSR}}$ | mm | Elevation of top face of SSR electrical enclosure. |
| $z\_{\text{bung}}$ \newline `z_bung` | $z\_{\text{drum,bot}} + z\_{\text{bung\_offset}}$ | mm | Elevation of open top rim of stainless steel dilution cup. |
| $x\_{\text{bung}}, y\_{\text{bung}}$ \newline `x_bung, y_bung` | $x\_k + r\_{\text{bung}} \cos\theta\_{\text{bung}}, \; y\_k + r\_{\text{bung}} \sin\theta\_{\text{bung}}$ | mm | Planar coordinates of exhaust dilution cup center. |
| $H\_{\text{shield}}$ \newline `h_shield` | $H\_{\text{wall}} - g\_{\text{floor}} - g\_{\text{ceil}}$ | mm | Vertical height of Promatect-H shield panels. |
| $z\_{\text{mirror}}$ \newline `z_mirror` | $2 z\_{\text{reflector,top}} - z\_{\text{drum,bot}}$ | mm | Elevation of virtual specular mirror image of kiln base in floor reflector. |
| $z\_{\text{hood,inlet}}$ \newline `z_hood_inlet` | $H\_{\text{wall}} - z\_{\text{hood\_drop}}$ | mm | Elevation of canopy hood capture mouth above screed ($z = 0$). |
| $d\_{\text{eff}}$ \newline `d_chimney_eff` | $d\_{\text{chimney}} (1 - f\_{\text{block}})$ | mm | Effective hydraulic depth of behind-shield chimney channel. |
| $U\_{\text{aac\_wall}}$ \newline `u_aac_wall` | $\left( R\_{si} + \frac{t\_{\text{aac}}}{k\_{\text{aac}}} + R\_{se} \right)^{-1}$ | $\text{W/m}^2\text{K}$ | Unshielded thermal transmittance of bare AAC block wall ($R\_{si}=0.13, R\_{se}=0.04$). |
| $(UA)\_{\text{upper}}$ \newline `compute_ua_upper_zone` | $U\_{\text{roof}} A\_{\text{roof}} + U\_{\text{aac}} A\_{\text{wall,up}} + U\_{\text{door}} A\_{\text{door,up}}$ | W/K | Parallel envelope thermal conductance of upper room gas layer above $z\_{\text{layer}}$. |
| $(UA)\_{\text{lower}}$ \newline `compute_ua_lower_zone` | $U\_{\text{aac}} A\_{\text{wall,low}} + U\_{\text{door}} A\_{\text{door,low}} + U\_{\text{subfloor}} A\_{\text{floor}}$ | W/K | Parallel envelope thermal conductance of lower room zone below $z\_{\text{layer}}$. |

---

## 4. Governing Physics & Mathematical Formulations

### 4.1 Dynamic Kiln Skin Energy Balance

Rather than fixing the kiln jacket temperature uniformly across all standoff distances, each cylindrical patch $m$, the top lid patch, and the bottom disk patch solve dynamically for their local outer surface temperature $T\_{\text{skin},m}$ by enforcing steady energy conservation between internal chamber conduction, external natural convection, and net radiative flux:

$$U\_{\text{kiln}} \left( T\_{\text{int}} - T\_{\text{skin},m} \right) = h\_{\text{kiln},m} \left( T\_{\text{skin},m} - T\_{\text{air,local}} \right) + q''\_{\text{rad,net},m}$$

where:
- Natural convection heat transfer coefficient $h\_{\text{kiln},m}$ is evaluated using Churchill-Chu correlations for vertical cylinders and laminar/turbulent correlations for horizontal upward and downward disks.
- $q''\_{\text{rad,net},m}$ is the net outgoing radiosity flux solving the enclosure radiosity system.
- Back-radiation from the hot corner shields reduces $q''\_{\text{rad,net},m}$ on corner-facing cylindrical patches, naturally reproducing the circumferential jacket temperature spread ($\Delta T\_{\text{skin}}$).

### 4.2 Patch Centroids, View Factor Algebra & Diffuse-Specular Radiosity

1. **Exact Surface Centroids:**
   - **Planar Rectangular Patches (Walls, Shields, Gypsum, SSR Faces):** The centroid is the exact vertex average:
     $$\vec{r}\_m = \left( \frac{x\_{\min} + x\_{\max}}{2}, \; \frac{y\_{\min} + y\_{\max}}{2}, \; \frac{z\_{\min} + z\_{\max}}{2} \right)$$
   - **Curved Kiln Jacket Patches:** Evaluated on the true cylindrical surface radius at midpoint angle $\theta\_c = (\theta\_0 + \theta\_1)/2$ and midpoint height $z\_c = (z\_0 + z\_1)/2$:
     $$\vec{r}\_m = \left( x\_k + R\_{\text{kiln}} \cos\theta\_c, \; y\_k + R\_{\text{kiln}} \sin\theta\_c, \; z\_c \right), \quad \vec{n}\_m = \left( \cos\theta\_c, \; \sin\theta\_c, \; 0 \right)$$
2. **Direct Diffuse View Factors ($F\_{m \to n}$):**
   - Computed via the double-area contour integral formula with numerical line-of-sight ray tracing to verify visibility against occlusion by the central kiln cylinder.
   - For coaxial parallel disks (kiln bottom plate disk to circular reflector disk separated by axial gap $h = z\_{\text{drum,bot}} - z\_{\text{reflector,top}}$), exact analytical view factors are used:
     $$F\_{1 \to 2} = \frac{1}{2} \left[ S - \sqrt{S^2 - 4 \left( \frac{r\_2}{r\_1} \right)^2} \right], \quad S = 1 + \frac{1 + (r\_2/h)^2}{(r\_1/h)^2}$$
3. **Specular Floor Reflector Virtual Image:**
   - Radiation reflecting specularly off the circular polished stainless steel reflector ($\varepsilon\_{\text{reflector}} = 0.15, \rho\_s = 0.85$) is modeled by constructing a virtual mirror-image kiln bottom disk at elevation:
     $$z\_{\text{mirror}} = 2 z\_{\text{reflector,top}} - z\_{\text{drum,bot}}$$
   - The effective view factor matrix combines direct diffuse exchange and mirror reflections:
     $$F\_{\text{eff},mn} = F\_{\text{direct},mn} + \rho\_s F\_{\text{mirror},mn}$$
   - Row conservation to the open ambient room enclosure satisfies:
     $$F\_{\text{room},m} = 1 - \sum\_n F\_{\text{eff},mn}$$
4. **Radiosity System Solve:**
   - The radiosity vector $J$ ($\text{W/m}^2$) solves the pre-factored linear system:
     $$\left[ I - (1 - \varepsilon) F\_{\text{eff}} \right] J = \varepsilon \sigma T^4 + (1 - \varepsilon) F\_{\text{room}} \sigma T\_{\text{room}}^4$$
   - Net outgoing radiative flux is computed as:
     $$q''\_{\text{rad,net},m} = \frac{\varepsilon\_m}{1 - \varepsilon\_m} \left( \sigma T\_m^4 - J\_m \right)$$

### 4.3 Behind-Shield Chimney Buoyant Channel Flow

The simulator resolves all three possible physical regimes in the behind-shield chimney:
1. **Flush Solid Contact ($d\_{\text{chimney}} \le 1\text{ mm}$):**
   - Natural draft velocity is zero ($v\_{\text{ch}} = 0$).
   - Heat transfers from the shield back face through the bare AAC block wall via direct 1D solid conduction in series with the exterior film resistance:
     $$q''\_{\text{back}} = \frac{T\_{\text{shield,back}} - T\_{\text{ext}}}{\frac{t\_{\text{aac}}}{k\_{\text{aac}}} + R\_{se}}$$
2. **Stagnant Closed Cavities ($g\_{\text{floor}} = 0$ or $g\_{\text{ceil}} = 0$):**
   - Natural draft velocity is zero ($v\_{\text{ch}} = 0$).
   - Heat transfers across the stagnant air gap via parallel conduction and radiative exchange:
     $$q''\_{\text{gap}} = \left( \frac{k\_{\text{air}}}{d\_{\text{eff}}} + h\_{\text{rad,gap}} \right) \left( T\_{\text{shield,back}} - T\_{\text{aac,front}} \right) = \frac{T\_{\text{aac,front}} - T\_{\text{ext}}}{\frac{t\_{\text{aac}}}{k\_{\text{aac}}} + R\_{se}}$$
3. **Open Buoyant Natural Draft Channel ($d\_{\text{chimney}} > 0, g\_{\text{floor}} > 0, g\_{\text{ceil}} > 0$):**
   - Hydrostatic buoyancy driving pressure balances inlet orifice turning loss, Darcy-Weisbach duct friction, and ceiling discharge loss:
     $$\Delta P\_{\text{buoy}} = \rho\_\infty g H\_{\text{shield}} \left( 1 - \frac{T\_{\text{lower}}}{\overline{T}\_{\text{ch}}} \right) = \frac{1}{2} \rho\_{\text{ch}} v\_{\text{ch}}^2 \left[ K\_{\text{in,eff}} + f\_{\text{D}} \frac{H\_{\text{shield}}}{2 d\_{\text{eff}}} + K\_{\text{out}}\left(\frac{d\_{\text{eff}}}{g\_{\text{ceil}}}\right) \right]$$
   - When `seal_chimney_sides = true`, air enters exclusively through $g\_{\text{floor}}$ ($K\_{\text{in,eff}} = 1.5 + 0.5 (d\_{\text{eff}} / g\_{\text{floor}})^2$).
   - When `seal_chimney_sides = false`, outer columns draw additional lateral room air through open vertical side slots.
   - Convective air heating and solid AAC conduction are solved stepwise upward along each column:
     $$\dot{m}'\_{\text{ch}} c\_p \frac{dT\_{\text{ch}}}{dz} = h\_{\text{ch}} \left( T\_{\text{shield,back}}(z) - T\_{\text{ch}}(z) \right) + h\_{\text{ch}} \left( T\_{\text{aac,front}}(z) - T\_{\text{ch}}(z) \right)$$

### 4.4 Floor Reflector, Under-Pan Air Gap, and Subfloor VIP Interface

To evaluate thermal protection of the subfloor VIP layer ($z = -t\_{\text{screed}}$) and surrounding vinyl ($z = 0$), the model solves three coupled floor zones:

1. **Center of Kiln / Under-Reflector Zone ($r \le D\_{\text{reflector}}/2$):**
   - The top surface of the reflector ($z = z\_{\text{reflector,top}}$) absorbs $(1 - \rho\_s) q''\_{\text{inc,base}}$ from the kiln bottom disk.
   - **Flush Reflector ($g\_{\text{floor\_reflector}} = 0\text{ mm}$):** Heat conducts directly through $t\_{\text{reflector\_calsil}}$, $t\_{\text{floor\_pad}}$, $t\_{\text{screed}}$, and $t\_{\text{vip}}/k\_{\text{vip}}$ in series to the structural slab ($T\_{\text{ext}}$).
   - **Air-Gapped Reflector ($g\_{\text{floor\_reflector}} > 0\text{ mm}$):** Natural convection ($h\_{\text{conv}} \approx 3.0\ \text{W/m}^2\text{K}$) and across-gap radiation cool the underside of the reflector pan to the lower room air stream, significantly reducing heat delivered into the floor pad below.
2. **Floor Pad Outside Reflector Rim:**
   - Evaluates point sensors on the exposed tile/pad along the corner diagonal (`pad_edge_corner`) and room diagonal (`pad_edge_room`), plus a dense radial candidate grid spanning the reflector optical shadow and penumbra up to the floor pad edge (`pad_outside_corner_*`, `pad_outside_room_*`).
   - Each sensor solves the local radiative-convective-conductive equilibrium:
     $$\varepsilon\_{\text{pad}} G\_{\text{inc}} - \varepsilon\_{\text{pad}} \sigma T\_{\text{pad}}^4 - h\_c (T\_{\text{pad}} - T\_{\text{lower}}) - \frac{T\_{\text{pad}} - T\_{\text{ext}}}{\frac{t\_{\text{floor\_pad}}}{k\_{\text{pad}}} + \frac{t\_{\text{screed}}}{k\_{\text{screed}}} + \frac{t\_{\text{vip}}}{k\_{\text{vip}}}} = 0$$
   - The corresponding peak VIP temperature directly below this point is evaluated via:
     $$T\_{\text{vip,outside}} = T\_{\text{ext}} + (T\_{\text{pad,edge,max}} - T\_{\text{ext}}) \frac{R\_{\text{vip}}}{R\_{\text{pad}} + R\_{\text{screed}} + R\_{\text{vip}}}$$
3. **Vinyl Edge Sensors:**
   - Evaluates sensors at $(W\_{\text{floor\_pad}} + 5\text{ mm}, y\_k, 0)$ and $(x\_k, W\_{\text{floor\_pad}} + 5\text{ mm}, 0)$ on the vinyl flooring resting directly on the screed, verifying compliance against the 45.0  deg C threshold.

### 4.5 Corner-Modified Plume & Active Hood Capture

1. **Total Convective Plume Heat Release:**
   $$\dot{Q}\_{\text{plume}} = \dot{Q}\_{\text{conv,cyl}} + \dot{Q}\_{\text{conv,lid}} + (1 - \text{bung}\_{\text{cap}}) \dot{Q}\_{\text{bung}}$$
2. **Virtual Origin & Standoff Image Multiplier ($m\_{\text{eff}}$):**
   - Virtual origin of the finite area source:
     $$z\_v = z\_{\text{kiln,top}} - 1.02 D\_{\text{kiln}} + 0.083 \left( \frac{\dot{Q}\_{\text{plume}}}{1000} \right)^{2/5}$$
   - Effective standoff distance to corner shields: $s\_{\text{eff}} = \min(s\_{\text{kiln,left}}, s\_{\text{kiln,back}}) + R\_{\text{kiln}}$.
   - Mowrer-Williamson corner image multiplier at hood height:
     $$m\_{\text{eff}} = 1 + 3 \exp\left( - \left[ \frac{s\_{\text{eff}}}{0.12 (z\_{\text{hood,inlet}} - z\_v)} \right]^2 \right)$$
3. **Plume Flow & Capture Radius at Hood Inlet:**
   - 95% Gaussian plume radius at hood mouth: $r\_{\text{plume,95}} = 1.65 (0.12 \Delta z\_{\text{plume}}) + d\_{\text{offset}}$.
   - Volumetric plume flow rate:
     $$\dot{V}\_{\text{plume}} = \frac{\dot{m}\_{\text{plume}}}{\rho\_{\text{plume}}}, \quad \dot{m}\_{\text{plume}} = 0.071 (\dot{Q}\_{\text{plume,kW}} m\_{\text{eff}})^{1/3} \Delta z\_{\text{plume}}^{5/3} m\_{\text{eff}}^{-1/3} + 0.018 \dot{Q}\_{\text{plume,kW}}^{1/3} D\_{\text{kiln}}$$
4. **Hood Capture Fraction ($\eta\_{\text{plume}}$) and Full Capture Flag:**
   - Geometric overlap fraction: $\eta\_{\text{geom}} = 1.0$ if $R\_{\text{hood}} \ge r\_{\text{plume,95}}$, else $1 - \exp(-0.5 (R\_{\text{hood}}/\sigma\_r)^2)$.
   - Volumetric capacity fraction: $\eta\_{\text{vol}} = \min(1.0, \dot{V}\_{\text{fan}} / \dot{V}\_{\text{plume}})$.
   - Combined capture fraction: $\eta\_{\text{plume}} = \eta\_{\text{geom}} \eta\_{\text{vol}}$.
   - Full capture boolean flag:
     $$\text{plume\_fully\_captured} = \left( \dot{V}\_{\text{fan}} \ge \dot{V}\_{\text{plume}} \right) \land \left( R\_{\text{hood}} \ge r\_{\text{plume,95}} \right)$$
5. **Behind-Shield Chimney Capture Fraction ($\eta\_{\text{chimney}}$):**
   - Modeled via surplus extraction capacity $\dot{V}\_{\text{surplus}} = \max(0, \dot{V}\_{\text{fan}} - \eta\_{\text{plume}} \dot{V}\_{\text{plume}})$, potential sink velocity $v\_{\text{sink}}$, and chimney vertical exit jet momentum.

### 4.6 Two-Zone Room Enclosure, Stratification & Ajar Doorway Balance

Convective heat from uncaptured plume $(1 - \eta\_{\text{plume}})\dot{Q}\_{\text{plume}}$, uncaptured chimney $(1 - \eta\_{\text{chimney}})\dot{Q}\_{\text{chimney}}$, and front wall surfaces heats the two-zone room enclosure:
1. **Stratification Interface Height ($z\_{\text{layer}}$):** Determined from doorway buoyancy neutral plane and spill rate $\dot{m}\_{\text{spill}}$.
2. **Doorway Buoyancy Exchange:**
   $$\dot{m}\_{\text{door}} = \frac{1}{2} \rho\_\infty W\_{\text{door\_open}} (H\_{\text{door}} - z\_{\text{layer}}) \cdot 0.65 \sqrt{g (H\_{\text{door}} - z\_{\text{layer}}) \frac{T\_{\text{upper}} - T\_{\text{ext}}}{T\_{\text{ext}}}}$$
3. **Coupled Zone Temperatures ($T\_{\text{room,upper}}, T\_{\text{room,lower}}$):**
   - Solved simultaneously by balancing heat inputs against envelope conductances $(UA)\_{\text{upper}}$, $(UA)\_{\text{lower}}$, and airflow advection through the doorway and hood fan.
4. **Ceiling Point Temperature Verification:**
   - Evaluates point sensors at the canopy hood outer rim (`ceiling_hood_rim_corner`, `ceiling_hood_rim_room`) and directly above the chimney exit (`ceiling_chimney_corner`), including uncaptured chimney jet mixing ($35\%$ local temperature elevation).

### 4.7 SSR Enclosure Lumped Thermal Model

The louvered SSR enclosure at $\theta\_{\text{SSR}} = -45^\circ$ exchanges radiation across its exposed outer rear and side faces with the kiln jacket, left shield, floor reflector, and room. Internally, air enters the bottom louvers driven by buoyancy from internal Joule heating $\dot{Q}\_{\text{SSR,int}} = 18.8\text{ W}$ and inward housing conduction, solving simultaneously for the outer casing temperature $T\_{\text{SSR,case}}$ and internal relay air temperature $T\_{\text{SSR,air}}$.

---

## 5. Solver Pipeline & Iteration Loops

```
[Stage 1: Input Loading & Geometry Setup]
   |-- Load .textproto, compute MD5 digest, validate physical bounds
   |-- Compute derived elevations (z_pad_top, z_reflector_bot, z_reflector_top, z_drum_bot, z_kiln_top, z_ssr_bot, z_bung, z_mirror)
   |-- Discretize SurfaceMesh:
   |      - Left & Back Shields (wall_id, col, row indices)
   |      - Exposed Gypsum Walls
   |      - Kiln Cylinder (on-skin cylindrical centroids), Lid, and Base disks
   |      - Floor Reflector disk
   |      - SSR Enclosure (rear and side vertical faces)
   |      - Floor sensors (pad edge, radial penumbra array, vinyl edges) and ceiling sensors

[Stage 2: Pre-Compute Geometry & View Factors]
   |-- Compute patch-to-patch diffuse view factor matrix F_direct with ray-traced cylinder occlusion
   |-- Compute exact coaxial parallel disks view factor for kiln bottom to reflector
   |-- Add specular reflector mirror-image matrix F_mirror (z_mirror) -> effective matrix F_eff
   |-- Compute sensor view factor vectors dF_{sensor -> n}
   |-- Pre-factor LU decomposition of radiosity matrix [I - (1 - eps) * F_eff]

[Stage 3 & 4: Coupled Thermal-Fluid Convergence Loop]
   |-- Initialize T_room_lower = T_ext, T_room_upper = T_ext + 2.0 K
   |-- OUTER LOOP (Room Stratification, Plume State & Ambient Air, max passes until delta_T < 0.05 K):
   |     |
   |     |-- INNER VECTORIZED NEWTON SOLVER (Surface Energy Balance):
   |     |     1. Solve linear radiosity system via LU back-substitution for J and q_rad_net
   |     |     2. Solve 1D behind-shield chimney flow, vertical air heating T_ch(z), and AAC conduction
   |     |     3. Evaluate Churchill-Chu and disk natural convection coefficients h_c(T)
   |     |     4. Compute exact analytical local derivatives dR_m / dT_m
   |     |     5. Execute damped Newton step on surface temperatures until max residual < 0.05 W/m^2
   |     |
   |     |-- Solve SSR internal air (T_ssr_air) and casing (T_ssr_case) temperatures
   |     |-- Solve corner-modified plume flow, hood capture fractions, doorway exchange, and room temps
   |     |-- Check outer loop convergence |T_upper_new - T_upper_old| < 0.05 K

[Stage 5: Post-Processing & Safety Metrics]
   |-- Solve floor sensor point temperatures (pad edge, outside VIP, vinyl edge)
   |-- Solve ceiling sensor point temperatures (hood rim, chimney exit)
   |-- Extract peak metrics, evaluate safety flags, and assemble SimulationOutput protobuf
```

---

## 6. Safety Pass/Fail Thresholds & Telemetry Outputs

Every simulation run evaluates and outputs the following telemetry metrics against conservative engineering safety limits:

| Telemetry Key | Physical Meaning | Pass Threshold | Failure Consequence if Exceeded |
|:------------------------------|:---------------------------------------|:---------------|:-------------------------------------------------------------------|
| `t_gypsum_max_left_c` | Peak surface temperature on exposed gypsum on Left Wall | $< 40.0^\circ\text{C}$ | Slow dehydration of $\text{CaSO}\_4 \cdot 2\text{H}\_2\text{O} \to$ hemihydrate (Siniat $40^\circ\text{C}$ / British Gypsum $49^\circ\text{C}$ limit). |
| `t_gypsum_max_back_c` | Peak surface temperature on exposed gypsum on Back Wall | $< 40.0^\circ\text{C}$ | Slow dehydration and structural degradation of gypsum board. |
| `gypsum_safe` | Boolean flag: true if both left and back exposed gypsum $< 40.0^\circ\text{C}$ | `true` | Unsafe thermal exposure of studio plasterboard. |
| `t_ssr_air_c` | Steady air temperature surrounding relay heatsink inside SSR enclosure | $< 35.0^\circ\text{C}$ | Exceeds Rohde ambient electronics rating ($30^\circ\text{C}-35^\circ\text{C}$); causes solid-state relay thermal derating or failure. |
| `t_ssr_case_c` | Area-averaged sheet-metal casing temperature of SSR box | $< 45.0^\circ\text{C}$ | Excessive radiative heating onto internal wiring and control circuitry. |
| `ssr_safe` | Boolean flag: true if `t_ssr_air_c < 35.0` and `t_ssr_case_c < 45.0` | `true` | Overheating of kiln firing control electronics. |
| `t_vip_under_reflector_c` | Peak temperature at top barrier film of subfloor VIP under kiln center | $< 50.0^\circ\text{C}$ | Accelerates gas permeation across the metallized polymer VIP envelope weld, permanently degrading insulation. |
| `t_floor_pad_under_reflector_c` | Temperature of floor topping pad under reflector center | Diagnostic ( deg C) | Quantifies under-reflector floor heating. |
| `floor_under_reflector_safe` | Boolean flag: true if `t_vip_under_reflector_c < 50.0` | `true` | Subfloor VIP degradation under kiln footprint. |
| `t_vip_outside_reflector_c` | Peak temperature of subfloor VIP outside reflector rim under pad edge | $< 50.0^\circ\text{C}$ | VIP thermal degradation in exposed floor zone outside reflector shadow. |
| `t_floor_pad_edge_c` | Peak temperature of concrete/tile floor pad outside reflector rim | Diagnostic ( deg C) | Indicates maximum surface temperature on exposed floor topping pad. |
| `t_vinyl_edge_c` | Peak temperature on industrial vinyl flooring outside square pad cutout | $< 45.0^\circ\text{C}$ | Vinyl plasticizer migration, softening, curling, or adhesive release. |
| `floor_outside_reflector_safe` | Boolean flag: true if `t_vip_outside_reflector_c < 50.0` and `t_vinyl_edge_c < 45.0` | `true` | Thermal damage to vinyl flooring or perimeter subfloor VIPs. |
| `floor_safe` | Boolean flag: true if both under-reflector and outside-reflector zones are safe | `true` | Overall subfloor and vinyl floor safety requirement. |
| `t_shield_max_c` | Peak front-face surface temperature on Promatect-H shield boards | $< 250.0^\circ\text{C}$ | Continuous operating service limit of Promatect-H calcium silicate. |
| `t_aac_max_c` | Peak front-face temperature of bare AAC block behind or above shield | $< 120.0^\circ\text{C}$ | Thermal drying shrinkage cracking threshold for aerated concrete masonry. |
| `v_chimney_max_m_s` | Maximum upward natural draft air velocity in behind-shield chimney | Diagnostic (m/s) | Indicates buoyant draft vigor in ventilated channel. |
| `t_chimney_exit_max_c` | Peak exit air temperature discharging from top of shield gap | Diagnostic ( deg C) | Quantifies warm air plume injected into ceiling layer from chimney. |
| `plume_fully_captured` | Boolean flag: true if hood fan CFM $\ge$ plume CFM AND hood radius $\ge r\_{\text{plume,95}}$ | `true` | Spillage of hot firing gases and fumes into studio room. |
| `plume_capture_fraction` | Fraction of primary kiln plume enthalpy captured by overhead hood | 0.0 to 1.0 | Quantitative hood capture efficiency metric. |
| `chimney_capture_fraction` | Fraction of behind-shield chimney exit flow drawn into overhead hood | 0.0 to 1.0 | Lateral draw capture of buoyant wall jet by hood blower. |
| `t_room_upper_c` | Steady-state air temperature of stratified upper room ceiling layer | Diagnostic ( deg C) | Upper studio comfort and thermal load. |
| `t_room_lower_c` | Steady-state air temperature of lower make-up air layer | Diagnostic ( deg C) | Working environment temperature at kiln operator height. |
| `z_stratification_layer_mm` | Elevation of thermal stratification layer interface above screed ($z = 0$) | Diagnostic (mm) | Height of warm ceiling smoke/gas layer. |
| `delta_t_kiln_skin_c` | Circumferential peak-to-minimum jacket temperature difference around kiln | Diagnostic ( deg C) | Quantifies back-radiation jacket heating asymmetry from corner shields. |
| `t_ceiling_max_c` | Peak surface temperature on overhead ceiling at $z = H\_{\text{wall}}$ | $< 40.0^\circ\text{C}$ | Thermal degradation of ceiling plasterboard above kiln and chimney. |
| `ceiling_safe` | Boolean flag: true if `t_ceiling_max_c < 40.0` | `true` | Ceiling fire/thermal protection limit. |
