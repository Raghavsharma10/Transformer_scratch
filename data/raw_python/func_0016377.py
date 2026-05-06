def reconcile_lines(lines1, lines2):
    """
    Return a dict `{lineno1: lineno2}` which reconciles line numbers `lineno1`
    of list `lines1` to line numbers `lineno2` of list `lines2`. Only lines
    that are common in both sets are present in the dict, lines unique to one
    of the sets are omitted.
    """
    differ = difflib.Differ()
    diff = differ.compare(lines1, lines2)

    SAME = '  '
    ADDED = '+ '
    REMOVED = '- '
    INFO = '? '

    lineno_map = {}  # {lineno1: lineno2, ...}
    lineno1_offset = 0
    lineno2 = 1

    for diffline in diff:
        if diffline.startswith(INFO):
            continue

        if diffline.startswith(SAME):
            lineno1 = lineno2 + lineno1_offset
            lineno_map[lineno1] = lineno2

        elif diffline.startswith(ADDED):
            lineno1_offset -= 1

        elif diffline.startswith(REMOVED):
            lineno1_offset += 1
            continue

        lineno2 += 1

    return lineno_map