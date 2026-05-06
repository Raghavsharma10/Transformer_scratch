def readProcess(cmd, *args):
    r"""Similar to `os.popen3`, but returns 2 strings (stdin, stdout) and the
    exit code (unlike popen2, exit is 0 if no problems occured (for some
    bizarre reason popen2 returns None... <sigh>).

    FIXME: only works for UNIX; handling of signalled processes.
    """
    import popen2
    BUFSIZE=1024
    import select
    popen = popen2.Popen3((cmd,) + args, capturestderr=True)
    which = {id(popen.fromchild): [],
             id(popen.childerr):  []}
    select = Result(select.select)
    read   = Result(os.read)
    try:
        popen.tochild.close() # XXX make sure we're not waiting forever
        while select([popen.fromchild, popen.childerr], [], []):
            readSomething = False
            for f in select.result[0]:
                while read(f.fileno(), BUFSIZE):
                    which[id(f)].append(read.result)
                    readSomething = True
            if not readSomething:
                break
        out, err = ["".join(which[id(f)])
                    for f in [popen.fromchild, popen.childerr]]
        returncode = popen.wait()

        if os.WIFEXITED(returncode):
            exit = os.WEXITSTATUS(returncode)
        else:
            exit = returncode or 1 # HACK: ensure non-zero
    finally:
        try:
            popen.fromchild.close()
        finally:
            popen.childerr.close()
    return out or "", err or "", exit