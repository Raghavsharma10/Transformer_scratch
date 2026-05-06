def captureOutput(func, *args, **kwargs):
    """
    Runs the specified function and arguments, and returns the
    tuple (stdout, stderr) as strings.
    """
    stdout = sys.stdout
    sys.stdout = StringIO.StringIO()
    stderr = sys.stderr
    sys.stderr = StringIO.StringIO()
    try:
        func(*args, **kwargs)
        stdoutOutput = sys.stdout.getvalue()
        stderrOutput = sys.stderr.getvalue()
    finally:
        sys.stdout.close()
        sys.stdout = stdout
        sys.stderr.close()
        sys.stderr = stderr
    return stdoutOutput, stderrOutput