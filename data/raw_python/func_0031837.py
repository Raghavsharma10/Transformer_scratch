def daemon_run(no_error, restart, record_path, keep_json, check_duplicate,
               use_polling, log_level):
    """
    Run RASH index daemon.

    This daemon watches the directory ``~/.config/rash/data/record``
    and translate the JSON files dumped by ``record`` command into
    sqlite3 DB at ``~/.config/rash/data/db.sqlite``.

    ``rash init`` will start RASH automatically by default.
    But there are alternative ways to start daemon.

    If you want to organize background process in one place such
    as supervisord_, it is good to add `--restart` option to force
    stop other daemon process if you accidentally started it in
    other place.  Here is an example of supervisord_ setup::

      [program:rash-daemon]
      command=rash daemon --restart

    .. _supervisord: http://supervisord.org/

    Alternatively, you can call ``rash index`` in cron job to
    avoid using daemon.  It is useful if you want to use RASH
    on NFS, as it looks like watchdog does not work on NFS.::

      # Refresh RASH DB every 10 minutes
      */10 * * * * rash index

    """
    # Probably it makes sense to use this daemon to provide search
    # API, so that this daemon is going to be the only process that
    # is connected to the DB?
    from .config import ConfigStore
    from .indexer import Indexer
    from .log import setup_daemon_log_file, LogForTheFuture
    from .watchrecord import watch_record, install_sigterm_handler

    install_sigterm_handler()
    cfstore = ConfigStore()
    if log_level:
        cfstore.daemon_log_level = log_level
    flogger = LogForTheFuture()

    # SOMEDAY: make PID checking/writing atomic if possible
    flogger.debug('Checking old PID file %r.', cfstore.daemon_pid_path)
    if os.path.exists(cfstore.daemon_pid_path):
        flogger.debug('Old PID file exists.  Reading from it.')
        with open(cfstore.daemon_pid_path, 'rt') as f:
            pid = int(f.read().strip())
        flogger.debug('Checking if old process with PID=%d is alive', pid)
        try:
            os.kill(pid, 0)     # check if `pid` is alive
        except OSError:
            flogger.info(
                'Process with PID=%d is already dead.  '
                'So just go on and use this daemon.', pid)
        else:
            if restart:
                flogger.info('Stopping old daemon with PID=%d.', pid)
                stop_running_daemon(cfstore, pid)
            else:
                message = ('There is already a running daemon (PID={0})!'
                           .format(pid))
                if no_error:
                    flogger.debug(message)
                    # FIXME: Setup log handler and flogger.dump().
                    # Note that using the default log file is not safe
                    # since it has already been used.
                    return
                else:
                    raise RuntimeError(message)
    else:
        flogger.debug('Daemon PID file %r does not exists.  '
                      'So just go on and use this daemon.',
                      cfstore.daemon_pid_path)

    with open(cfstore.daemon_pid_path, 'w') as f:
        f.write(str(os.getpid()))

    try:
        setup_daemon_log_file(cfstore)
        flogger.dump()
        indexer = Indexer(cfstore, check_duplicate, keep_json, record_path)
        indexer.index_all()
        watch_record(indexer, use_polling)
    finally:
        os.remove(cfstore.daemon_pid_path)