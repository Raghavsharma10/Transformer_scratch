def untranslateName(s):
    """Undo Python conversion of CL parameter or variable name."""

    s = s.replace('DOT', '.')
    s = s.replace('DOLLAR', '$')
    # delete 'PY' at start of name components
    if s[:2] == 'PY': s = s[2:]
    s = s.replace('.PY', '.')
    return s