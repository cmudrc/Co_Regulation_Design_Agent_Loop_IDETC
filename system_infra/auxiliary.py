from datetime import datetime
from typing import List, Dict

def generate_timestamped_basename(prefix=""):
    """Generate a shared basename using timestamp for HTML and JSON files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    basename = f"{timestamp}"
    # basename = f"{prefix}_{timestamp}"
    return basename

def truncate_messages(
    messages: List[Dict[str, str]],
    keep_initial: int = 2,
    keep_recent_turns: int = 5,
) -> List[Dict[str, str]]:
    """Truncate message history to simulate context window overflow.

    Preserves the first `keep_initial` messages (anchors: system prompt, initial
    user instructions, etc.) and the most recent `keep_recent_turns` user+assistant
    turn pairs. Middle messages are dropped.

    Args:
        messages: Full conversation history as list of role/content dicts.
        keep_initial: Number of leading messages to always preserve (default: 2,
                      i.e. system prompt + first user message).
        keep_recent_turns: Number of most recent user+assistant pairs to keep
                           beyond the anchor messages (default: 3).

    Returns:
        Truncated message list. If the history is short enough to fit within
        keep_initial + keep_recent_turns * 2, it is returned unchanged.
    """
    anchors = messages[:keep_initial]
    remainder = messages[keep_initial:]

    max_tail = keep_recent_turns * 2  # each turn = 1 user + 1 assistant message
    if len(remainder) <= max_tail:
        return messages  # nothing to drop

    tail = remainder[-max_tail:]
    dropped = len(remainder) - max_tail
    print(f"[INFO] Context truncation: dropped {dropped} middle message(s), "
          f"keeping {keep_initial} anchor(s) + {len(tail)} recent message(s).")
    return anchors + tail
