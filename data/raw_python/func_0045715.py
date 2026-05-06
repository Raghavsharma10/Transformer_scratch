def _delete_horizontal_space(text, pos):
    """Delete all spaces and tabs around pos."""
    while pos > 0 and text[pos - 1].isspace():
        pos -= 1
    end_pos = pos
    while end_pos < len(text) and text[end_pos].isspace():
        end_pos += 1
    return text[:pos] + text[end_pos:], pos