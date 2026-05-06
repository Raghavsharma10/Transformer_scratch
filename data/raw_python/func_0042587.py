def runCommandReturnOutput(cmd):
    """
    Runs a shell command and return the stdout and stderr
    """
    splits = shlex.split(cmd)
    proc = subprocess.Popen(
        splits, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(stdout, stderr)
    return stdout, stderr