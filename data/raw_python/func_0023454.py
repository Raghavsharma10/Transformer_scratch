def spawn_gdb(pid, address=DFLT_ADDRESS, gdb='gdb', verbose=False,
              ctx=None, proc_iut=None):
    """Spawn gdb and attach to a process."""

    parent, child = socket.socketpair()
    proc = Popen([gdb, '--interpreter=mi', '-nx'],
                    bufsize=0, stdin=child, stdout=child, stderr=STDOUT)
    child.close()

    connections = {}
    gdb = GdbSocket(ctx, address, proc, proc_iut, parent, verbose,
                    connections)
    gdb.mi_command('-target-attach %d' % pid)
    gdb.cli_command('python import pdb_clone.bootstrappdb_gdb')
    asyncore.loop(map=connections)
    proc.wait()
    return gdb.error