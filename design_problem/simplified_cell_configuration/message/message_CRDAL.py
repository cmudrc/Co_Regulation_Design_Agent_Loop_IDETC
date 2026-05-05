from design_problem_infra.progress_analyzer import analyze_progress

_DESIGN_INSTRUCTIONS = """\
    Generate a battery pack design using 18650 cells to satisfy all constraints, while maximizing capacity.
    The required voltage is the target voltage, the required capacity is the minimum capacity, and the required current is the minimum current.
    The required temperature is the maximum allowable temperature. The required dimensions are the maximum allowable dimensions.

    Assume 18650 cells have a nominal voltage of 3.7V, a nominal capacity of 2.5Ah, and an internal resistance of 0.05Ohm.
    The 18650 cells have a diameter of 18mm and a height of 65mm.
    The cells have to be placed in an upright orientation (cylindrical axis vertical), in a grid pattern, and with uniform spacing between cells.
    The minimum spacing between cells is 2mm to allow for cooling and manufacturing tolerances. Assume 20 degrees Celsius ambient temperature. 

    You MUST output ALL THREE of the following fields using this exact syntax:
    - CELL_LOCATIONS: [[x1, y1, z1], [x2, y2, z2], ...]
    - CELL_CONNECTIONS: [number_of_series, number_of_parallel]
    - CELL_SPACING: [spacing_in_mm]

    Where each [x, y, z] is the grid position (integers) of a cell in the pack. All coordinates must be non-negative integers.
    The coordinate system originates at [0, 0, 0]. Coordinate [0, 0, 0] does not necessarily need to be occupied, but it is the origin point for the design.
    The battery pack design uses hexagonal close packing. Your CELL_LOCATIONS will be converted to actual physical positions automatically.
    You don't need to worry about how the cells are packed or physically arranged, just focus on the grid coordinates and the spacing.
    The CELL_SPACING is the spacing in millimeters between adjacent cells, only integer values.

    Format each design step as:
    <think>
    [Analysis of current state and reasoning for next modification]
    </think>
    <action>
    CELL_LOCATIONS: [Grammar action]
    CELL_CONNECTIONS: [Grammar action]
    CELL_SPACING: [Grammar action]
    </action>

    IMPORTANT: After each step, you will receive feedback from the supervisor and the battery pack simulator showing the actual performance results of your designs. Use this feedback to guide your next design iteration.

    TERMINATION RULES:
    - Only output <answer> AFTER you have received feedback from the supervisor and the simulator, and the design is valid (Design status: FEASIBLE).
    - Only output <answer> when you are confident the design cannot be meaningfully improved further. You can receive multiple rounds of feedback showing FEASIBLE status before you decide to terminate.

    Once the design meets termination goals (after receiving feedback showing FEASIBLE status), end with:
    <think>
    [Final assessment of the optimized design and summary of the optimization process]
    </think>
    <action>
    CELL_LOCATIONS: [Grammar action]
    CELL_CONNECTIONS: [Grammar action]
    CELL_SPACING: [Grammar action]
    </action>
    <answer>
    Final capacity: [value]Ah
    </answer>"""


def design_agent_initial_message(problem_query, step):

    problem_prompt = f"""<problem>{problem_query}</problem>

{_DESIGN_INSTRUCTIONS}
    """

    return problem_prompt


def design_agent_follow_up_message(feedback, step):

    if step.design_dict is not None:
        feedback_prompt = f"\n\n<feedback>"
        feedback_prompt += f"\nPerformance of your current design:"

        feedback_prompt += f"\nCapacity: {step.design_dict.get('design_capacity', 0):.1f}"

        feedback_prompt += f"\nx-dimension: {step.design_dict.get('design_width', 0):.1f}"
        feedback_prompt += f"\ny-dimension: {step.design_dict.get('design_depth', 0):.1f}"
        feedback_prompt += f"\nz-dimension: {step.design_dict.get('design_height', 0):.1f}"

        feedback_prompt += f"\nVoltage: {step.design_dict.get('design_voltage', 0):.1f}"
        feedback_prompt += f"\nMax current: {step.design_dict.get('max_current', 0):.1f}"
        feedback_prompt += f"\nMax temperature: {step.design_dict.get('max_temperature_C', 0):.1f}C"
        feedback_prompt += f"\nSeries count: {step.design_dict.get('series_count', 0):.0f}"
        feedback_prompt += f"\nParallel count: {step.design_dict.get('parallel_count', 0):.0f}"
        feedback_prompt += f"\nCell spacing: {step.design_dict.get('cell_spacing', 0):.0f}mm"

        feedback_prompt += f"\nDesign status: {'FEASIBLE' if step.get('is_valid', False) else 'INFEASIBLE'}"

    else:
        feedback_prompt = f"\n\n<feedback>"
        feedback_prompt += f"\nNo specific feedback as the design failed to parse. Please check your response format. "

    # Collect all errors from validation results
    all_errors = []
    if step.validation_results:
        for validator_name, result in step.validation_results.items():
            if hasattr(result, 'errors') and result.errors:
                all_errors.extend(result.errors)

    if all_errors:
        feedback_prompt += f"\n\nWARNING: {len(all_errors)} errors detected:"
        for error in all_errors:
            feedback_prompt += f"\n  - {error}"

    feedback_prompt += f"\n\nSupervisor feedback: {feedback}"

    feedback_prompt += f"""\n\nGenerate the next design iteration.
    If you believe the current design is satisfactory (Design status: FEASIBLE and the design cannot be meaningfully improved further), output
    <think>
    [Final assessment of the optimized design and summary of the optimization process]
    </think>
    <action>
    CELL_LOCATIONS: [Grammar action]
    CELL_CONNECTIONS: [Grammar action]
    CELL_SPACING: [Grammar action]
    </action>
    <answer>
    Final capacity: [value]Ah
    </answer>
    to terminate.
    """

    feedback_prompt += f"\n</feedback>"

    return feedback_prompt


def CRDAL_agent_initial_message(problem_query):
    initial_prompt = f"""
    A designer is working on the following battery pack design problem:

    <problem>{problem_query}</problem>

    The designer is maximizing capacity while satisfying all constraints. And the designer received the following instructions:

    <designer_instructions>
{_DESIGN_INSTRUCTIONS}
    </designer_instructions>

    You will receive the designer's performance metrics after each iteration.
    Your job is to monitor the optimization trajectory — track progress trends, detect stalls or regressions, and suggest strategic shifts.

    The designer may be forgetful, so you need to remind them of the design objectives and their progress.
    The designer may also be fixated, so you need to remind them to explore the design space, try potential alternatives, and pursuit higher performance designs. 
    """

    return initial_prompt


def CRDAL_agent_follow_up_message(step, step_history):
    """Build a history-aware, structured prompt for the supervisor.

    Args:
        step: Current DesignStep (just completed by design agent)
        step_history: List of all previous DesignSteps (chronological)
    """

    # ── Current design summary ──
    if step.design_dict is not None:
        user_query = f"\nPerformance of Current Design (Step {step.step_number}):"

        user_query += f"\nCapacity: {step.design_dict.get('design_capacity', 0):.1f}"

        user_query += f"\nx-dimension: {step.design_dict.get('design_width', 0):.1f}"
        user_query += f"\ny-dimension: {step.design_dict.get('design_depth', 0):.1f}"
        user_query += f"\nz-dimension: {step.design_dict.get('design_height', 0):.1f}"

        user_query += f"\nVoltage: {step.design_dict.get('design_voltage', 0):.1f}"
        user_query += f"\nMax current: {step.design_dict.get('max_current', 0):.1f}"
        user_query += f"\nMax temperature: {step.design_dict.get('max_temperature_C', 0):.1f}C"
        user_query += f"\nSeries count: {step.design_dict.get('series_count', 0):.0f}"
        user_query += f"\nParallel count: {step.design_dict.get('parallel_count', 0):.0f}"
        user_query += f"\nCell spacing: {step.design_dict.get('cell_spacing', 0):.0f}mm"

        user_query += f"\nDesign status: {'FEASIBLE' if step.get('is_valid', False) else 'INFEASIBLE'}"

    else:
        user_query = f"\nPerformance of Current Design (Step {step.step_number}):"
        user_query += "Design failed to parse (grammar error). No metrics available.\n"

    # Collect errors
    all_errors = []
    if step.validation_results:
        for validator_name, result in step.validation_results.items():
            if hasattr(result, 'errors') and result.errors:
                all_errors.extend(result.errors)

    if all_errors:
        user_query += f"Errors ({len(all_errors)}):\n"
        for error in all_errors:
            user_query += f"  - {error}\n"

    # ── Progress trajectory + trend summary ──
    user_query += analyze_progress(step_history)

    # ── Structured response request ──
    user_query += """
== Your Task ==
Based on the trajectory above, respond in this exact format:

PROGRESS_ASSESSMENT: [improving / stalling / regressing]
BIGGEST_BOTTLENECK: [which metric or constraint is the main issue right now]
RECOMMENDED_ACTION: [one specific strategic suggestion for the designer's next iteration]
"""

    return user_query

