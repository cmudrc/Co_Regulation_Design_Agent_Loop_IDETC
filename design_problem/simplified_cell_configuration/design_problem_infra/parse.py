import ast
import re

### Extract Design Information from Model Response

def _extract_action_text(text):
    """Extract content from the first <action>...</action> block.
    Returns None if no <action> block is found.
    """
    match = re.search(r'<action>(.*?)</action>', text, re.DOTALL | re.IGNORECASE)
    if not match:
        print("[ERROR] No <action> block found in response.")
        return None
    return match.group(1)


def extract_list_block(field_name, text):
    """
    Extracts a bracketed list from the text for a given field name.
    Handles optional prefixes (-, *), any casing, and multiline brackets.
    """
    # Match field name with optional prefix and any casing
    pattern = rf"(?:[-*]?\s*)?{field_name}\s*[:=]\s*\["
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        print(f"[ERROR] Start of {field_name} block not found.")
        return None

    # Walk forward from the matched '[' to find balanced brackets
    start = match.end() - 1
    bracket_count = 1
    end = start
    while end < len(text) - 1 and bracket_count > 0:
        end += 1
        if text[end] == "[":
            bracket_count += 1
        elif text[end] == "]":
            bracket_count -= 1

    try:
        raw_block = text[start:end + 1]
        return ast.literal_eval(raw_block)
    except Exception as e:
        print(f"[ERROR] Failed to parse {field_name}: {e}")
        return None

def extract_scalar(field_name, text):
    """Extract a numeric scalar value for field_name from text."""
    pattern = rf"{field_name}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
    m = re.search(pattern, text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    print(f"[ERROR] {field_name} not found or invalid.")
    return None


def extract_cell_locations(text):
    action_text = _extract_action_text(text)
    if action_text is None:
        return None
    result = extract_list_block("CELL_LOCATIONS", action_text)
    if result is None:
        return None
    # Filter out non-list elements (e.g., Ellipsis from "..." in LLM output)
    # Also filter entries where any coordinate value is non-numeric (e.g., Ellipsis inside a coordinate)
    filtered = [
        loc for loc in result
        if isinstance(loc, (list, tuple)) and all(isinstance(v, (int, float)) for v in loc)
    ]
    if not filtered:
        print("[ERROR] CELL_LOCATIONS contains no valid coordinate entries")
        return None
    return filtered

# def extract_cell_connections(text):
#     return extract_list_block("CELL_CONNECTIONS", text)

def extract_cell_connections(text):
    action_text = _extract_action_text(text)
    if action_text is None:
        return None, None
    result = extract_list_block("CELL_CONNECTIONS", action_text)
    if result is None or len(result) != 2:
        print("[ERROR] CELL_CONNECTIONS not found or invalid format")
        return None, None

    number_of_series = result[0]
    number_of_parallel = result[1]
    return number_of_series, number_of_parallel


def extract_cell_spacing(text):
    action_text = _extract_action_text(text)
    if action_text is None:
        return None
    result = extract_list_block("CELL_SPACING", action_text)
    if result is None or len(result) != 1:
        print("[ERROR] CELL_SPACING not found or invalid format")
        return None
    value = result[0]
    if not isinstance(value, (int, float)):
        print(f"[ERROR] CELL_SPACING value is not numeric: {value!r}")
        return None
    return int(value)
