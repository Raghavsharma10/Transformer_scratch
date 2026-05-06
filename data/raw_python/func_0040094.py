def linkorcopy(src, dst):
    """Hardlink src file to dst if possible, otherwise copy."""
    if not os.path.isfile(src):
        raise error.ButcherError('linkorcopy called with non-file source. '
                                 '(src: %s  dst: %s)' % src, dst)
    elif os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    elif os.path.exists(dst):
        os.unlink(dst)
    elif not os.path.exists(os.path.dirname(dst)):
        os.makedirs(os.path.dirname(dst))

    try:
        os.link(src, dst)
        log.debug('Hardlinked: %s -> %s', src, dst)
    except OSError:
        shutil.copy2(src, dst)
        log.debug('Couldn\'t hardlink. Copied: %s -> %s', src, dst)