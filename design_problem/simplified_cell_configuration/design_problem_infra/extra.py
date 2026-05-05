import random
import math
import numpy as np
import plotly.graph_objects as go

def generate_random_prompt(seed=None, cell_limit_LB=4, cell_limit_UB=10):
    """Generate a random battery pack design prompt with specified seed."""
    if seed is not None:
        random.seed(seed)

    # Random features
    series_options = [i for i in range(cell_limit_LB, cell_limit_UB + 1)] # number of cells in series
    parallel_options = [i for i in range(cell_limit_LB, cell_limit_UB + 1)] # number of cells in parallel

    series_count = random.choice(series_options)
    parallel_count = random.choice(parallel_options)

    voltage = round(series_count * 3.7, 1)
    capacity = round(parallel_count * 2.5, 1)
    current = round(parallel_count * 15, 1)
    temperature_C = random.choice([50, 60, 70, 80])

    width = 10 + 20 * max(series_count, parallel_count) * 5
    depth = 10 + 20 * max(series_count, parallel_count) * 5
    height = 10 + 65 * max(series_count, parallel_count) * 2

    query = (
        f"Design a {voltage}V battery pack "
        f"with a minimum capacity of {capacity} Ah, "
        f"capable of continuously supplying at least {current}A of current draw "
        f"while staying at or below {temperature_C}C during operation, "
        f"within a {width}mm × {depth}mm × {height}mm envelope."
    )
    required_specs = {
                "voltage": voltage,
                "capacity": capacity,
                "current": current,
                "temperature_C": temperature_C,
                "width_mm": width,
                "depth_mm": depth,
                "height_mm": height,
            }

    return query, required_specs

def render_battery_pack(
    cell_locations,
    cell_spacing: float = 2.0,
    output_html: str = "battery_pack_rendering.html",
    open_browser: bool = False,
):
    """
    Render a 3D battery pack using a hexagonal close-packed (HCP) layout.

    Cells are vertical 18650 cylinders (axis along z).  In the x-y plane each
    row is offset by half the horizontal pitch so that cells nest into the gaps
    of the row below, minimising dead space (HCP).

    Layout geometry
    ---------------
    pitch_x  = diameter + cell_spacing          (centre-to-centre, same row)
    pitch_y  = pitch_x * sqrt(3)/2              (centre-to-centre, adjacent rows)
    offset_x = pitch_x / 2                      (odd rows shifted right by this)
    pitch_z  = cell_length + cell_spacing        (layer-to-layer)

    Parameters
    ----------
    cell_locations : list of [x, y, z] or [x, y, z, present]
        Integer grid coordinates.  If a 4th element is present it is treated as
        a boolean presence flag (0 = absent, 1 = present).
    cell_spacing : float
        Gap in mm between adjacent cell surfaces.  Default 2.0 mm.
    output_html : str
        Output filename for the interactive HTML file.
    open_browser : bool
        Whether to automatically open the file in the browser.
    """
    DIAMETER   = 18.0  # mm  (18650 cell)
    RADIUS     = DIAMETER / 2.0
    CELL_LEN   = 65.0  # mm
    RESOLUTION = 30

    pitch_x  = DIAMETER + cell_spacing
    pitch_y  = pitch_x * (math.sqrt(3) / 2.0)
    offset_x = pitch_x / 2.0
    pitch_z  = CELL_LEN + cell_spacing

    def _is_present(loc):
        return len(loc) < 4 or loc[3] == 1

    def _grid_to_mm(gx, gy, gz):
        """Convert integer grid coordinates to mm using HCP geometry."""
        cx = gx * pitch_x + (gy % 2) * offset_x
        cy = gy * pitch_y
        cz = gz * pitch_z
        return cx, cy, cz

    def _cell_surfaces(x0, y0, z0):
        theta = np.linspace(0, 2 * np.pi, RESOLUTION)

        # Side surface
        theta_grid, z_grid = np.meshgrid(theta, [0, CELL_LEN])
        xs = RADIUS * np.cos(theta_grid) + x0
        ys = RADIUS * np.sin(theta_grid) + y0
        zs = z_grid + z0

        # Top and bottom caps
        r_grid, th_grid = np.meshgrid(np.linspace(0, RADIUS, 2), theta)
        xc = r_grid * np.cos(th_grid) + x0
        yc = r_grid * np.sin(th_grid) + y0
        zt = np.full_like(xc, z0 + CELL_LEN)
        zb = np.full_like(xc, z0)

        return (xs, ys, zs), (xc, yc, zt), (xc, yc, zb)

    fig = go.Figure()

    for loc in cell_locations:
        if not _is_present(loc):
            continue
        gx, gy, gz = int(loc[0]), int(loc[1]), int(loc[2])
        cx, cy, cz = _grid_to_mm(gx, gy, gz)
        (xs, ys, zs), (xt, yt, zt), (xb, yb, zb) = _cell_surfaces(cx, cy, cz)

        kwargs = dict(colorscale="Blues", showscale=False, hoverinfo="skip")
        fig.add_trace(go.Surface(x=xs, y=ys, z=zs, **kwargs))  # side
        fig.add_trace(go.Surface(x=xt, y=yt, z=zt, **kwargs))  # top cap
        fig.add_trace(go.Surface(x=xb, y=yb, z=zb, **kwargs))  # bottom cap

    fig.update_layout(
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
        ),
        title="3D Battery Pack — HCP Layout (18650 cells, vertical)",
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )

    fig.write_html(output_html, auto_open=open_browser)
    print(f"Saved: {output_html}")


if __name__ == "__main__":
    # Sample: 6 wide × 4 deep × 2 high, all cells present
    W, D, H = 6, 4, 2
    locations = [
        [x, y, z]
        for z in range(H)
        for y in range(D)
        for x in range(W)
    ]
    render_battery_pack(
        cell_locations=locations,
        cell_spacing=2.0,
        output_html="battery_pack_6x4x2_hcp.html",
        open_browser=True,
    )
