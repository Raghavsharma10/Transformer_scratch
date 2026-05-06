def packtar(tarfile, files, srcdir):
    """ Pack the given files into a tar, setting cwd = srcdir"""
    nullfd = open(os.devnull, "w")
    tarfile = cygpath(os.path.abspath(tarfile))
    log.debug("pack tar %s from folder  %s with files ", tarfile, srcdir)
    log.debug(files)
    try:
        check_call([TAR, '-czf', tarfile] + files, cwd=srcdir,
                   stdout=nullfd, preexec_fn=_noumask)
    except Exception:
        log.exception("Error packing tar file %s to %s", tarfile, srcdir)
        raise
    nullfd.close()