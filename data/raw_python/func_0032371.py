def invocation():
    """reconstructs the invocation for this python program"""
    cmdargs = [sys.executable] + sys.argv[:]
    invocation = " ".join(shlex.quote(s) for s in cmdargs)
    return invocation