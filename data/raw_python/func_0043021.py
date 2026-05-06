def get_revision():
    """
    GET THE CURRENT GIT REVISION
    """
    proc = Process("git log", ["git", "log", "-1"])

    try:
        while True:
            line = proc.stdout.pop().strip().decode('utf8')
            if not line:
                continue
            if line.startswith("commit "):
                return line[7:]
    finally:
        with suppress_exception:
            proc.join()