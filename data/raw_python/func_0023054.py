def find_program_variables(code):
    """
    Return a dict describing program variables::

        {'var_name': ('uniform|attribute|varying', type), ...}

    """
    vars = {}
    lines = code.split('\n')
    for line in lines:
        m = re.match(r"\s*" + re_prog_var_declaration + r"\s*(=|;)", line)
        if m is not None:
            vtype, dtype, names = m.groups()[:3]
            for name in names.split(','):
                vars[name.strip()] = (vtype, dtype)
    return vars