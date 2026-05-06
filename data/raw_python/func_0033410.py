def clear_stale_pids(pids, pid_dir='/tmp', prefix='', multi=False):
    'check for and remove any pids which have no corresponding process'
    if isinstance(pids, (int, float, long)):
        pids = [pids]
    pids = str2list(pids, map_=unicode)
    procs = map(unicode, os.listdir('/proc'))
    running = [pid for pid in pids if pid in procs]
    logger.warn(
        "Found %s pids running: %s" % (len(running),
                                       running))
    prefix = prefix.rstrip('.') if prefix else None
    for pid in pids:
        if prefix:
            _prefix = prefix
        else:
            _prefix = unicode(pid)
        # remove non-running procs
        if pid in running:
            continue
        if multi:
            pid_file = '%s%s.pid' % (_prefix, pid)
        else:
            pid_file = '%s.pid' % (_prefix)
        path = os.path.join(pid_dir, pid_file)
        if os.path.exists(path):
            logger.debug("Removing pidfile: %s" % path)
            try:
                remove_file(path)
            except OSError as e:
                logger.warn(e)
    return running