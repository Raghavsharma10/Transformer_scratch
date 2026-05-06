def get_output(cmd, args):
    """Runs a command and returns its output (stdout + stderr).

    :param str|unicode cmd:
    :param str|unicode|list[str|unicode] args:

    :rtype: str|unicode

    """
    from subprocess import Popen, STDOUT, PIPE

    command = [cmd]
    command.extend(listify(args))

    process = Popen(command, stdout=PIPE, stderr=STDOUT)
    out, _ = process.communicate()

    return out.decode('utf-8')