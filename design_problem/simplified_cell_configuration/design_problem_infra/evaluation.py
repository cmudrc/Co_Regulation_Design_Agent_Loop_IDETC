import re
import math
import numpy as np
from scipy.optimize import brentq
from typing import Dict, List, Optional, Any

# 18650 Cell specifications
CELL_SPECS = {
    "nominal_voltage": 3.7,
    "max_voltage": 4.2,
    "min_voltage": 2.5,
    "nominal_capacity": 2.5,
    "weight": 0.045,
    "diameter": 18.0,
    "length": 65.0,
    "internal_resistance": 0.05,
    "max_discharge_rate": 10,
    "max_charge_rate": 2,
    "cost": 10
}

# Design constraints - USER CONFIGURABLE
MIN_SPACING = 2.0  # mm between cells (minimum allowed)
SAFETY_MARGIN = 0.0  # mm between cells and pack wall

# Thermal simulation parameters
h_conv = 10.0  # W/(m^2*K) convective heat transfer coefficient
h_rad = 0.5  # emissivity for radiative heat transfer
stefan_boltzmann = 5.670374419e-8  # W/(m^2*K^4) Stefan-Boltzmann constant
ambient_temp_C = 20.0  # C ambient temperature


def spacing_geometry(cell_spacing: float):
    """Compute spacing-dependent geometry constants from a given cell_spacing (mm)."""

    cell_radius = CELL_SPECS["diameter"] / 2.0
    cell_spacing_h = CELL_SPECS["diameter"] + cell_spacing
    cell_spacing_v = CELL_SPECS["length"] + cell_spacing
    hex_x_spacing = cell_spacing_h
    hex_y_spacing = cell_spacing_h * (math.sqrt(3) / 2.0)
    hex_offset_x = cell_spacing_h / 2.0
    return cell_radius, cell_spacing_h, cell_spacing_v, hex_x_spacing, hex_y_spacing, hex_offset_x


### Extract Design Features

def extract_design_dimensions(cell_locations: List[List[int]], 
                              cell_spacing: float) -> Dict[str, Any]:
    """
    Extract grid dimensions from cell locations.

    Args:
        cell_locations: List of [x, y, z] or [x, y, z, present] coordinates
        cell_spacing: Spacing between cells in mm

    Returns:
        Dict with keys: num_cells_width, num_cells_depth, num_cells_height, pack_width, pack_depth, pack_height
    """
    if not cell_locations:
        return {"num_cells_width": 0, "num_cells_depth": 0, "num_cells_height": 0,
                "design_width": 0, "design_depth": 0, "design_height": 0}

    # Filter for present cells (if 4th element exists, check if it's 1)
    active_cells = []
    for loc in cell_locations:
        if len(loc) >= 4:
            if loc[3] == 1:  # present flag
                active_cells.append(loc[:3])
        elif len(loc) == 3:
            active_cells.append(loc[:3])
        elif len(loc) == 2:
            active_cells.append([loc[0], loc[1], 0])  # pad missing z as single layer

    if not active_cells:
        return {"num_cells_width": 0, "num_cells_depth": 0, "num_cells_height": 0,
                "design_width": 0, "design_depth": 0, "design_height": 0}

    cells = np.array(active_cells)

    cell_radius, cell_spacing_h, cell_spacing_v, hex_x_spacing, hex_y_spacing, hex_offset_x = spacing_geometry(cell_spacing)

    # Find max values (grid) in each dimension (add 1 because coordinates are 0-indexed)
    num_cells_width = int(np.max(cells[:, 0])) + 1
    num_cells_depth = int(np.max(cells[:, 1])) + 1
    num_cells_height = int(np.max(cells[:, 2])) + 1

    # Find max values (physical mm) in each dimension
    max_mm_width = (num_cells_width - 1) * hex_x_spacing + CELL_SPECS["diameter"] + 2 * SAFETY_MARGIN + hex_offset_x # adjust as the last cell has no MIN_SPACING
    max_mm_depth = (num_cells_depth - 1) * hex_y_spacing + CELL_SPECS["diameter"] + 2 * SAFETY_MARGIN # adjust as the last cell has no MIN_SPACING
    max_mm_height = (num_cells_height - 1) * cell_spacing_v + CELL_SPECS["length"] + 2 * SAFETY_MARGIN

    return {
        "num_cells_width": num_cells_width,
        "num_cells_depth": num_cells_depth,
        "num_cells_height": num_cells_height,
        "design_width": max_mm_width,
        "design_depth": max_mm_depth,
        "design_height": max_mm_height
    }


def extract_mechanical_specs(cell_locations: List[List[int]],
                             cell_spacing: float,
                             number_of_parallel: int = 1,
                             required_current: float = 0.0,) -> Dict[str, Any]:
    """
    Extract mechanical specs.
    """

    dimensions = extract_design_dimensions(cell_locations, cell_spacing)
    design_width = dimensions["design_width"]
    design_depth = dimensions["design_depth"]
    design_height = dimensions["design_height"]

    # count actual cells
    cell_count = 0
    for loc in cell_locations:
        if len(loc) >= 4:
            if loc[3] == 1:  # present flag
                cell_count += 1
        else:
            cell_count += 1

    # calculate surface area in mm^2
    surface_area = 2 * (design_width * design_depth + design_width * design_height + design_depth * design_height)

    # calculate volume in mm^3
    design_volume = design_width * design_depth * design_height

    # Estimate max temperature (estimated to be roughly = surface temperature) at required minimum current.
    # Q_joule = cell_count × I_per_cell^2 × R_cell
    # Solve heat balance: Q = h_conv·A·(Ts−Ta) + ε·σ·A·(Ts^4−Ta^4)
    I_per_cell = required_current / max(number_of_parallel, 1)
    Q_joule = cell_count * (I_per_cell ** 2) * CELL_SPECS["internal_resistance"]
    A_m2 = surface_area * 1e-6  # mm^2 → m^2
    T_amb_K = ambient_temp_C + 273.15
    if Q_joule == 0.0:
        max_temperature_C = ambient_temp_C
    else:
        try:
            T_max_K = brentq(
                lambda T_s: (h_conv * A_m2 * (T_s - T_amb_K)
                             + h_rad * stefan_boltzmann * A_m2 * (T_s ** 4 - T_amb_K ** 4)
                             - Q_joule),
                T_amb_K, T_amb_K + 500.0
            )
            max_temperature_C = T_max_K - 273.15
        except ValueError:
            max_temperature_C = ambient_temp_C + 500.0  # heat cannot be dissipated within range

    return {
        "cell_count": cell_count,
        "surface_area": surface_area,
        "design_volume": design_volume,
        "max_temperature_C": max_temperature_C,
    }


def extract_electrical_specs(cell_locations: List[List[int]],
                             cell_spacing: float,
                             number_of_series: int,
                             number_of_parallel: int,
                             required_temperature_C: float = ambient_temp_C) -> Dict[str, Any]:
    """
    Extract electrical specs.
    """

    mechanical_specs = extract_mechanical_specs(cell_locations, cell_spacing)
    cell_count = mechanical_specs["cell_count"]
    surface_area = mechanical_specs["surface_area"]

    # calculate expected voltage (series affects voltage) and capacity (parallel affects capacity)
    nominal_voltage = CELL_SPECS["nominal_voltage"]
    actual_voltage = number_of_series * nominal_voltage

    # calculate expected capacity (parallel affects capacity)
    nominal_capacity = CELL_SPECS["nominal_capacity"]
    actual_capacity = number_of_parallel * nominal_capacity

    # calculate maximum current at the required maximum temperature
    # Q_joule = cell_count × I_per_cell^2 × R_cell
    # Solve heat balance: Q = h_conv·A·(Ts−Ta) + ε·σ·A·(Ts^4−Ta^4)

    A_m2 = surface_area * 1e-6  # mm^2 → m^2
    T_amb_K = ambient_temp_C + 273.15
    T_max_K = required_temperature_C + 273.15
    Q_joule = h_conv * A_m2 * (T_max_K - T_amb_K) + h_rad * stefan_boltzmann * A_m2 * (T_max_K ** 4 - T_amb_K ** 4)

    if cell_count == 0:
        return {"design_voltage": 0, "design_capacity": 0, "max_current": 0}
    max_thermal_per_cell = Q_joule / cell_count
    sqrt_arg = max_thermal_per_cell / CELL_SPECS["internal_resistance"]
    max_current_per_cell = math.sqrt(max(0.0, sqrt_arg))
    max_current = max_current_per_cell * number_of_parallel

    return {
        "design_voltage": actual_voltage, 
        "design_capacity": actual_capacity,
        "max_current": max_current
    }


def extract_design_features(
        cell_locations: List[List[int]],
        cell_spacing: float,
        number_of_series: int,
        number_of_parallel: int,
        required_current: float = 0.0,
        required_temperature_C: float = ambient_temp_C) -> Dict[str, Any]:

    """
    output:
        {
        "cell_locations": cell_locations
        "cell_spacing": cell_spacing,

        "num_cells_width": num_cells_width,
        "num_cells_depth": num_cells_depth,
        "num_cells_height": num_cells_height,
        "design_width": max_mm_width,
        "design_depth": max_mm_depth,
        "design_height": max_mm_height,

        "cell_count": cell_count,
        "surface_area": surface_area,
        "design_volume": design_volume,
        "max_temperature_C": max_temperature_C,

        "series_count": number_of_series,
        "parallel_count": number_of_parallel,

        "design_voltage": actual_voltage,
        "design_capacity": actual_capacity,
        "max_current": max_current
        }
    """

    design_cell_locations = {"cell_locations": cell_locations, "cell_spacing": cell_spacing}
    design_dimension_dict = extract_design_dimensions(cell_locations, cell_spacing)
    cell_count_dict = extract_mechanical_specs(cell_locations, cell_spacing, number_of_parallel, required_current)
    design_cell_connections = {"series_count": number_of_series, "parallel_count": number_of_parallel}
    electrical_specs_dict = extract_electrical_specs(cell_locations, cell_spacing, number_of_series, number_of_parallel, required_temperature_C)

    design_dict = design_cell_locations | design_dimension_dict | cell_count_dict | design_cell_connections | electrical_specs_dict

    return design_dict


def detect_termination(text: str) -> bool:
    """
    Check if the model response contains an <answer> tag,
    indicating it considers the design complete.
    """
    return bool(re.search(r"<answer>", text, re.IGNORECASE))