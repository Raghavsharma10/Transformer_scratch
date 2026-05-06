def _restore_case(s, memory):
    """Restore a lowercase string's characters to their original case."""
    cased_s = []
    for i, c in enumerate(s):
        if i + 1 > len(memory):
            break
        cased_s.append(c if memory[i] else c.upper())
    return ''.join(cased_s)