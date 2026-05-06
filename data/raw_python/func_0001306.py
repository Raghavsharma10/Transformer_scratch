def execute_command(command=None):
    """Execute a command and return the stdout and stderr."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stdin = process.communicate()
    process.wait()
    return (stdout, stdin), process.returncode