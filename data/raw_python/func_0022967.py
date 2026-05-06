def _get_dpi_from(cmd, pattern, func):
    """Match pattern against the output of func, passing the results as
    floats to func.  If anything fails, return None.
    """
    try:
        out, _ = run_subprocess([cmd])
    except (OSError, CalledProcessError):
        pass
    else:
        match = re.search(pattern, out)
        if match:
            return func(*map(float, match.groups()))