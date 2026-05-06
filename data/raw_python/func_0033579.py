def ida_spawn(ida_binary, filename, port=18861, mode='oneshot',
              processor_type=None, logfile=None):
    """
    Open IDA on the the file we want to analyse.

    :param ida_binary:  The binary name or path to ida
    :param filename:    The filename to open in IDA
    :param port:        The port on which to serve rpc from ida
    :param mode:        The server mode. "oneshot" to close ida when the connection is closed, or
                        "threaded" to run IDA visible to the user and allow multiple connections
    :param processor_type:
                        Which processor IDA should analyze this binary as, e.g. "metapc". If not
                        provided, IDA will guess.
    :param logfile:     The file to log IDA's output to. Default /tmp/idalink-{port}.log
    """
    ida_progname = _which(ida_binary)
    if ida_progname is None:
        raise IDALinkError('Could not find executable %s' % ida_binary)

    if mode not in ('oneshot', 'threaded'):
        raise ValueError("Bad mode %s" % mode)

    if logfile is None:
        logfile = LOGFILE.format(port=port)

    ida_realpath = os.path.expanduser(ida_progname)
    file_realpath = os.path.realpath(os.path.expanduser(filename))
    server_script = os.path.join(MODULE_DIR, 'server.py')

    LOG.info('Launching IDA (%s) on %s, listening on port %d, logging to %s',
             ida_realpath, file_realpath, port, logfile)

    env = dict(os.environ)
    if mode == 'oneshot':
        env['TVHEADLESS'] = '1'

    if sys.platform == "darwin":
        # If we are running in a virtual environment, which we should, we need
        # to insert the python lib into the launched process in order for IDA
        # to not default back to the Apple-installed python because of the use
        # of paths in library identifiers on macOS.
        if "VIRTUAL_ENV" in os.environ:
            env['DYLD_INSERT_LIBRARIES'] = os.environ['VIRTUAL_ENV'] + '/.Python'

    # The parameters are:
    # -A     Automatic mode
    # -S     Run a script (our server script)
    # -L     Log all output to our logfile
    # -p     Set the processor type

    command = [
        ida_realpath,
        '-A',
        '-S%s %d %s' % (server_script, port, mode),
        '-L%s' % logfile,
    ]
    if processor_type is not None:
        command.append('-p%s' % processor_type)
    command.append(file_realpath)

    LOG.debug('IDA command is %s', ' '.join("%s" % s for s in command))
    return subprocess.Popen(command, env=env)