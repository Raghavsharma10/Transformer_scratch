def unpackmar(marfile, destdir):
    """Unpack marfile into destdir"""
    marfile = cygpath(os.path.abspath(marfile))
    nullfd = open(os.devnull, "w")
    try:
        check_call([MAR, '-x', marfile], cwd=destdir,
                   stdout=nullfd, preexec_fn=_noumask)
    except Exception:
        log.exception("Error unpacking mar file %s to %s", marfile, destdir)
        raise
    nullfd.close()