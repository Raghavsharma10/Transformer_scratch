def generate_session_id(data):
    """
    Generate session ID based on HOST, TTY, PID [#]_ and start time.

    :type data: dict
    :rtype: str

    .. [#] PID of the shell, i.e., PPID of this Python process.

    """
    host = data['environ']['HOST']
    tty = data['environ'].get('TTY') or 'NO_TTY'
    return ':'.join(map(str, [
        host, tty, os.getppid(), data['start']]))