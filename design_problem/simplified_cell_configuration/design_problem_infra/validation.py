import numpy as np
from typing import Dict, List, Optional, Any
import pybamm
from .parse import extract_cell_locations, extract_cell_connections, extract_cell_spacing
from .evaluation import extract_design_features, ambient_temp_C


class ValidationResult:
    """Store validation results with detailed error messages."""

    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.details = {}

    def add_error(self, message: str):
        """Add an error message and mark validation as failed."""
        self.is_valid = False
        self.errors.append(message)

    def add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)

    def add_detail(self, key: str, value):
        """Add detail information."""
        self.details[key] = value

    def __str__(self):
        """Return a formatted string representation."""
        lines = []
        lines.append(f"Validation {'PASSED' if self.is_valid else 'FAILED'}")

        if self.errors:
            lines.append("\nErrors:")
            for error in self.errors:
                lines.append(f"  ❌ {error}")

        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  ⚠️  {warning}")

        if self.details:
            lines.append("\nDetails:")
            for key, value in self.details.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)


### Validations

# def validate_with_pybamm(features):
#     try:
#         model = pybamm.lithium_ion.SPM()
#         parameter_values = model.default_parameter_values
        
#         parameter_values.update({
#             "Nominal cell capacity [A.h]": features["generated_capacity"],
#             "Number of cells connected in series to make a battery": features["series_count"],
#             "Number of electrodes connected in parallel to make a cell": features["parallel_count"],
#         })
        
#         sim = pybamm.Simulation(model, parameter_values=parameter_values)
#         sim.solve([0, 600])
#         return True
    
#     except Exception as e:
#         print(f"❌ PyBaMM validation failed: {e}")
#         return False
    

def validate_cell_spacing(cell_spacing: float) -> ValidationResult:
    """Validate that cell spacing meets the minimum requirement."""
    result = ValidationResult()
    if cell_spacing < 2.0:
        result.add_error(
            f"Cell spacing too small: {cell_spacing}mm < minimum 2.0mm"
        )
    else:
        result.add_detail("cell_spacing_check", f"✓ {cell_spacing}mm >= minimum 2.0mm")
    return result


def validate_cell_locations(
    cell_locations: List[List[int]]
    ) -> ValidationResult:
    """
    Validate that cell locations are physically valid and there are no duplicate cells. 
    """

    result = ValidationResult()

    if not cell_locations:
        result.add_error("Cell locations list is empty")
        return result

    # Filter for present cells (if 4th element exists, check if it's 1)
    active_cells = []
    for loc in cell_locations:
        if len(loc) >= 4:
            if loc[3] == 1:  # present flag
                active_cells.append(loc[:3])
        else:
            active_cells.append(loc[:3])

    if not active_cells:
        result.add_error("No active cells found in cell_locations")
        return result

    # Convert to numpy array for easier processing
    cells = np.array(active_cells)

    # Check for negative coordinates (invalid grid positions)
    if np.any(cells < 0):
        neg_coords = [(i, list(active_cells[i])) for i in range(len(active_cells)) if any(v < 0 for v in active_cells[i])]
        result.add_error(
            f"Cell locations contain negative coordinates (invalid grid positions): "
            f"{neg_coords[:5]}{'...' if len(neg_coords) > 5 else ''}"
        )
        return result

    # Check for duplicate cells (same x, y, z coordinates)
    unique_cells = np.unique(cells, axis=0)
    if len(unique_cells) < len(cells):
        duplicates = len(cells) - len(unique_cells)
        result.add_error(f"Found {duplicates} duplicate cell location(s)")
    else:
        result.add_detail("cell_locations", f"✓ no duplicate cell locations found")

    return result


def validate_cell_count(
    cell_count: int,
    series_count: int,
    parallel_count: int
) -> ValidationResult:
    """
    Validate that actual cell count matches claimed series/parallel configuration.
    """

    result = ValidationResult()

    # Expected count from series/parallel
    expected_count = series_count * parallel_count
    result.add_detail("expected_count_from_sp", f"{series_count} × {parallel_count} = {expected_count}")

    if cell_count != expected_count:
        result.add_error(
            f"Cell count mismatch: found {cell_count} cells, but design claimed {series_count}S{parallel_count}P "
            f"claimed configuration requires {expected_count} cells"
        )
    else:
        result.add_detail("cell_count_check", f"✓ {cell_count} cells matches {series_count}S{parallel_count}P")

    return result


def validate_prompt_satisfaction(
    required_specs: Dict[str, float],
    design_width: float,
    design_depth: float,
    design_height: float,
    design_voltage: float,
    design_capacity: float,
    design_current: float,
    design_max_temperature_C: float,
    voltage_tolerance: float = 0.1
    ) -> ValidationResult:
    """
    Validate that generated design satisfies the prompt requirements.
    """

    result = ValidationResult()

    # Check voltage - should be exact match (within tolerance)
    req_voltage = required_specs.get("voltage")

    if req_voltage is not None and design_voltage is not None:
        voltage_diff = abs(design_voltage - req_voltage)
        if voltage_diff > voltage_tolerance:
            result.add_error(
                f"Voltage mismatch: required {req_voltage:.1f}V, generated {design_voltage:.1f}V "
                f"(difference: {voltage_diff:.1f}V, tolerance: {voltage_tolerance:.1f}V)"
            )
        else:
            result.add_detail("voltage_check", f"✓ generated {design_voltage:.1f}V matches required {req_voltage:.1f}V")
    elif req_voltage is not None and design_voltage is None:
        result.add_error("Voltage values missing for comparison")

    # Check capacity - generated should be >= required
    req_capacity = required_specs.get("capacity")

    if req_capacity is not None and design_capacity is not None:
        if design_capacity < req_capacity:
            result.add_error(
                f"Capacity insufficient: required >={req_capacity:.1f}Ah, generated {design_capacity:.1f}Ah "
                f"(shortfall: {req_capacity - design_capacity:.1f}Ah)"
            )
        else:
            result.add_detail("capacity_check", f"✓ generated {design_capacity:.1f}Ah >= required {req_capacity:.1f}Ah")
    elif req_capacity is not None and design_capacity is None:
        result.add_error("Capacity values missing for comparison")

    # Check current - generated should be >= required
    req_current = required_specs.get("current")

    if req_current is not None and design_current is not None:
        if design_current < req_current:
            result.add_error(
                f"Max current insufficient: required >={req_current:.1f}A, generated {design_current:.1f}A "
                f"(shortfall: {req_current - design_current:.1f}A)"
            )
        else:
            result.add_detail("max_current_check", f"✓ generated {design_current:.1f}A >= required {req_current:.1f}A")
    elif req_current is not None and design_current is None:
        result.add_error("Max current values missing for comparison")

    # Check temperature - generated should be <= required maximum
    req_temperature = required_specs.get("temperature_C")

    if req_temperature is not None and design_max_temperature_C is not None:
        if design_max_temperature_C > req_temperature:
            result.add_error(
                f"Temperature exceeds limit: required <={req_temperature:.1f}°C, "
                f"estimated {design_max_temperature_C:.1f}°C "
                f"(excess: {design_max_temperature_C - req_temperature:.1f}°C)"
            )
        else:
            result.add_detail("temperature_check", f"✓ estimated {design_max_temperature_C:.1f}°C <= required {req_temperature:.1f}°C")
    elif req_temperature is not None and design_max_temperature_C is None:
        result.add_error("Temperature values missing for comparison")

    # Check dimensions - generated should be <= required
    dimensions = [
        ("width_mm", "Width"),
        ("depth_mm", "Depth"),
        ("height_mm", "Height")
    ]
    generated_specs = {"width_mm": design_width, 
                       "depth_mm": design_depth,
                       "height_mm": design_height}

    for dim_key, dim_name in dimensions:
        req_dim = required_specs.get(dim_key)
        gen_dim = generated_specs.get(dim_key)

        if req_dim is not None and gen_dim is not None:
            if gen_dim > req_dim:
                result.add_error(
                    f"{dim_name} exceeds limit: required <={req_dim}mm, generated {gen_dim}mm "
                    f"(excess: {gen_dim - req_dim:.2f}mm)"
                )
            else:
                result.add_detail(f"{dim_key}_check", f"✓ generated {gen_dim}mm <= required {req_dim}mm")
        else:
            result.add_error(f"{dim_name} values missing for comparison")

    return result


# Validate design

def validate_design(
        text: str,
        required_specs: Optional[Dict] = None
):
    """
    Comprehensive validation of a battery pack design.

    Returns:
        design_dict: Dictionary with parsed design features
        validation_results: Dictionary with validation results for each check
    """

    validation_results = {}

    cell_locations = extract_cell_locations(text)
    number_of_series, number_of_parallel = extract_cell_connections(text)
    cell_spacing = extract_cell_spacing(text)

    # Check if extraction was successful
    if not cell_locations:
        error_result = ValidationResult()
        error_result.add_error("Failed to extract cell_locations from output")
        validation_results["extraction_error"] = error_result
        design_dict=None
        return design_dict, validation_results

    if number_of_series is None or number_of_parallel is None:
        error_result = ValidationResult()
        error_result.add_error("Failed to extract CELL_CONNECTIONS from output")
        validation_results["extraction_error"] = error_result
        design_dict=None
        return design_dict, validation_results

    if cell_spacing is None:
        error_result = ValidationResult()
        error_result.add_error("Failed to extract CELL_SPACING from output")
        validation_results["extraction_error"] = error_result
        return None, validation_results

    # Early check for negative coordinates before any calculations
    def _pad_loc(loc):
        if len(loc) >= 3:
            return loc[:3]
        elif len(loc) == 2:
            return [loc[0], loc[1], 0]
        elif len(loc) == 1:
            return [loc[0], 0, 0]
        else:
            return [0, 0, 0]

    active_for_check = [_pad_loc(loc)
                        for loc in cell_locations
                        if not (len(loc) >= 4 and loc[3] == 0)]
    if any(v < 0 for loc in active_for_check for v in loc):
        error_result = ValidationResult()
        error_result.add_error("Cell locations contain negative coordinates (not a valid design)")
        print("[ERROR] Cell locations contain negative coordinates — skipping design evaluation")
        validation_results["extraction_error"] = error_result
        return None, validation_results

    required_current = required_specs.get("current", 0.0) if required_specs else 0.0
    required_temperature_C = required_specs.get("temperature_C", ambient_temp_C) if required_specs else ambient_temp_C
    design_dict = extract_design_features(
        cell_locations=cell_locations, 
        cell_spacing=cell_spacing, 
        number_of_series=number_of_series, 
        number_of_parallel=number_of_parallel,
        required_current=required_current,
        required_temperature_C=required_temperature_C,
    )

    if not all(design_dict.values()):
        error_result = ValidationResult()
        missing_fields = [k for k, v in design_dict.items() if v is None]
        error_result.add_error(f"Failed to extract required features: {', '.join(missing_fields)}")
        validation_results["extraction_error"] = error_result
        design_dict=None
        return design_dict, validation_results

    # 1. validate_cell_spacing
    validation_results["cell_spacing_validity"] = validate_cell_spacing(cell_spacing)

    # 2. validate_cell_locations
    validation_results["cell_location_validity"] = validate_cell_locations(
        cell_locations=cell_locations
    )

    # 3. validate_cell_count
    validation_results["cell_count_validity"] = validate_cell_count(
        cell_count=int(design_dict["cell_count"]),
        series_count=int(design_dict["series_count"]),
        parallel_count=int(design_dict["parallel_count"])
    )

    # 4. validate_prompt_satisfaction
    if required_specs:
        validation_results["prompt_satisfaction_validity"] = validate_prompt_satisfaction(
            required_specs=required_specs,
            design_width=float(design_dict["design_width"]),
            design_depth=float(design_dict["design_depth"]),
            design_height=float(design_dict["design_height"]),
            design_voltage=float(design_dict["design_voltage"]),
            design_capacity=float(design_dict["design_capacity"]),
            design_current=float(design_dict["max_current"]),
            design_max_temperature_C=float(design_dict["max_temperature_C"]),
            )

    return design_dict, validation_results


def print_validation_summary(results: Dict[str, ValidationResult]) -> bool:
    """
    Print a comprehensive summary of all validation results.
    """
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")

    all_valid = all(result.is_valid for result in results.values())

    for check_name, result in results.items():
        print(f"\n[{check_name.upper().replace('_', ' ')}]")
        print(result)

    print("\n" + "="*50)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED")
        return True
    else:
        print("❌ VALIDATION FAILED")
        failed_checks = [name for name, result in results.items() if not result.is_valid]
        print(f"Failed checks: {', '.join(failed_checks)}")
        return False


