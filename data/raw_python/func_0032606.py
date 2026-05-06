def packmar(marfile, srcdir):
    """Create marfile from the contents of srcdir"""
    nullfd = open(os.devnull, "w")
    files = [f[len(srcdir) + 1:] for f in findfiles(srcdir)]
    marfile = cygpath(os.path.abspath(marfile))
    try:
        check_call(
            [MAR, '-c', marfile] + files, cwd=srcdir, preexec_fn=_noumask)
    except Exception:
        log.exception("Error packing mar file %s from %s", marfile, srcdir)
        raise
    nullfd.close()