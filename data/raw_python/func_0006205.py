def __find_executables(path):
    """Used by find_graphviz

    path - single directory as a string

    If any of the executables are found, it will return a dictionary
    containing the program names as keys and their paths as values.

    Otherwise returns None
    """

    success = False
    progs = {
        "dot": "",
        "twopi": "",
        "neato": "",
        "circo": "",
        "fdp": "",
        "sfdp": "",
    }

    was_quoted = False
    path = path.strip()
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
        was_quoted = True

    if not os.path.isdir(path):
        return None

    for prg in progs:
        if progs[prg]:
            continue

        prg_path = os.path.join(path, prg)
        prg_exe_path = prg_path + ".exe"

        if os.path.exists(prg_path):
            if was_quoted:
                prg_path = "\"{}\"".format(prg_path)
            progs[prg] = prg_path
            success = True

        elif os.path.exists(prg_exe_path):
            if was_quoted:
                prg_exe_path = "\"{}\"".format(prg_exe_path)
            progs[prg] = prg_exe_path
            success = True

    if success:
        return progs

    return None