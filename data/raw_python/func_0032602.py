def unpackexe(exefile, destdir):
    """Unpack the given exefile into destdir, using 7z"""
    nullfd = open(os.devnull, "w")
    exefile = cygpath(os.path.abspath(exefile))
    try:
        check_call([SEVENZIP, 'x', exefile], cwd=destdir,
                   stdout=nullfd, preexec_fn=_noumask)
    except Exception:
        log.exception("Error unpacking exe %s to %s", exefile, destdir)
        raise
    nullfd.close()