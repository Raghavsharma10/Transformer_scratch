def unpacktar(tarfile, destdir):
    """ Unpack given tarball into the specified dir """
    nullfd = open(os.devnull, "w")
    tarfile = cygpath(os.path.abspath(tarfile))
    log.debug("unpack tar %s into %s", tarfile, destdir)
    try:
        check_call([TAR, '-xzf', tarfile], cwd=destdir,
                   stdout=nullfd, preexec_fn=_noumask)
    except Exception:
        log.exception("Error unpacking tar file %s to %s", tarfile, destdir)
        raise
    nullfd.close()