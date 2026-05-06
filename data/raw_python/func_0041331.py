def run_command(command):
    """
    Utility function for run command with subprocess. Return a tuple, with
    return code and if python >= 2.7 command's output or None if python <= 2.6
    """
    try:
        check_ouput = getattr(
            subprocess, 'check_output', subprocess.check_call)
        result = check_ouput(command, stderr=subprocess.STDOUT, shell=True)
        if check_ouput.__name__ == 'check_output':
            return 0, result
        else:
            return result, None
    except subprocess.CalledProcessError as e:
        return e.returncode, getattr(e, 'output', None)