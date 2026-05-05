
def analyze_progress(step_history):
    """Generate a textual summary of the design progress trajectory.

    Args:
        step_history: List of DesignStep objects in chronological order.

    Returns:
        A string containing the progress trajectory table and trend summary.
    """
    query = ""

    # Filter to steps with valid design_dict for analysis
    valid_history = [s for s in step_history if s.design_dict is not None]

    if not valid_history:
        return query

    query += "\n== Progress Trajectory ==\n"
    query += "| Step | Capacity | Spacing | Valid | Change |\n"
    query += "|------|----------|---------|-------|--------|\n"

    prev = None
    flat_streak = 0  # consecutive steps with < 1% improvement in capacity

    for s in valid_history:
        d = s.design_dict
        cap = d.get('design_capacity', 0)
        spacing = d.get('cell_spacing', 0)
        valid = s.get('is_valid', False)

        # Compute change description (higher capacity = improvement)
        if prev is None:
            change = "-"
        else:
            prev_d = prev.design_dict
            old_cap = prev_d.get('design_capacity', 0)
            if old_cap > 0:
                pct = (cap - old_cap) / old_cap * 100
                if abs(pct) >= 1.0:
                    change = f"Cap {pct:+.1f}%"
                else:
                    change = "~flat"
                flat_streak = flat_streak + 1 if abs(pct) < 1.0 else 0
            else:
                change = "-"

        query += f"| {s.step_number:<4} | {cap:<8.1f} | {spacing:<7.0f} | {str(valid):<5} | {change} |\n"
        prev = s

    # ── Trend summary ──
    query += "\n== Trend Summary ==\n"

    if len(valid_history) >= 2:
        first = valid_history[0].design_dict
        last = valid_history[-1].design_dict

        first_cap = first.get('design_capacity', 0)
        last_cap = last.get('design_capacity', 0)
        if first_cap > 0:
            total_pct = (last_cap - first_cap) / first_cap * 100
            query += f"- Capacity: {total_pct:+.1f}% overall ({first_cap:.1f} -> {last_cap:.1f} Ah)\n"

        if flat_streak >= 2:
            query += f"- Stall detected: no meaningful capacity improvement (>1%) for {flat_streak} consecutive steps\n"

        # Check for regression (last step vs second-to-last)
        second_last = valid_history[-2].design_dict
        old_cap = second_last.get('design_capacity', 0)
        new_cap = last.get('design_capacity', 0)
        if old_cap > 0 and new_cap < old_cap * 0.99:
            query += f"- Regression detected in last step: Capacity {(new_cap - old_cap) / old_cap * 100:.1f}%\n"
    else:
        query += "- Only one valid design so far, no trend data yet.\n"

    return query



    #     user_query += """
    # == Your Task ==
    # Based on the trajectory above, respond in this exact format:

    # PROGRESS_ASSESSMENT: [improving / stalling / regressing]
    # BIGGEST_BOTTLENECK: [which metric or constraint is the main issue right now]
    # RECOMMENDED_ACTION: [one specific strategic suggestion for the designer's next iteration]
    # WHAT_TO_KEEP: [what is working well and should not change]
    # CONFIDENCE: [low / medium / high] - [one sentence justification]
    # """