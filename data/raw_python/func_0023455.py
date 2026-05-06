def attach_loop(argv):
    """Spawn the process, then repeatedly attach to the process."""

    # Check if the pdbhandler module is built into python.
    p = Popen((sys.executable, '-X', 'pdbhandler', '-c',
                'import pdbhandler; pdbhandler.get_handler().host'),
               stdout=PIPE, stderr=STDOUT)
    p.wait()
    use_xoption = True if p.returncode == 0 else False

    # Spawn the process.
    args = [sys.executable]
    if use_xoption:
        # Use SIGUSR2 as faulthandler is set on python test suite with
        # SIGUSR1.
        args.extend(['-X', 'pdbhandler=localhost 7935 %d' % signal.SIGUSR2])
        args.extend(argv)
        proc = Popen(args)
    else:
        args.extend(argv)
        proc = Popen(args)

    # Repeatedly attach to the process using the '-X' python option or gdb.
    ctx = Context()
    error = None
    time.sleep(.5 + random.random())
    while not error and proc.poll() is None:
        if use_xoption:
            os.kill(proc.pid, signal.SIGUSR2)
            connections = {}
            dev_null = io.StringIO() if PY3 else StringIO.StringIO()
            asock = AttachSocketWithDetach(connections, stdout=dev_null)
            asock.create_socket(socket.AF_INET, socket.SOCK_STREAM)
            connect_process(asock, ctx, proc)
            asyncore.loop(map=connections)
        else:
            error = spawn_gdb(proc.pid, ctx=ctx, proc_iut=proc)
        time.sleep(random.random())

    if error and gdb_terminated(error):
        error = None
    if proc.poll() is None:
        proc.terminate()
    else:
        print('pdb-attach: program under test return code:', proc.wait())

    result = str(ctx.result)
    if result:
        print(result)
    return error