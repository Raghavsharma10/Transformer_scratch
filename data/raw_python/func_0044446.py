def run_cmd(args, stream=False, shell=True):
    """
    Execute a command via :py:class:`subprocess.Popen`; return its output
    (string, combined STDOUT and STDERR) and exit code (int). If stream is True,
    also stream the output to STDOUT in realtime.

    :param args: the command to run and arguments, as a list or string
    :param stream: whether or not to stream combined OUT and ERR in realtime
    :type stream: bool
    :param shell: whether or not to execute the command through the shell
    :type shell: bool
    :return: 2-tuple of (combined output (str), return code (int))
    :rtype: tuple
    """
    s = ''
    if stream:
        s = ' and streaming output'
    logger.info('Running command%s: %s', s, args)
    outbuf = ''
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         shell=shell)
    logger.debug('Started process; pid=%s', p.pid)
    for c in iter(lambda: p.stdout.read(1), ''):
        outbuf += c
        if stream:
            sys.stdout.write(c)
    p.poll()  # set returncode
    logger.info('Command exited with code %d', p.returncode)
    logger.debug("Command output:\n%s", outbuf)
    return outbuf, p.returncode