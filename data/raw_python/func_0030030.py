def spawnPythonProcess(processProtocol, args=(), env={},
                       path=None, uid=None, gid=None, usePTY=0,
                       packages=()):
    """Launch a Python process

    All arguments as to spawnProcess(), except the executable
    argument is omitted.
    """
    return spawnProcess(processProtocol, sys.executable,
                        args, env, path, uid, gid, usePTY,
                        packages)