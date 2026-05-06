def port_in_use(port, kill=False, logging=False):
    """
    Checks whether a port is free or not.
    
    :param int port:
        The port number to check for.
        
    :param bool kill:
        If ``True`` the process will be killed.
    
    :returns:
        The process id as :class:`int` if in use, otherwise ``False`` .
    """

    command_template = 'lsof -iTCP:{0} -sTCP:LISTEN'
    process = subprocess.Popen(command_template.format(port).split(),
                               stdout=subprocess.PIPE)
    headers = process.stdout.readline().decode().split()

    if 'PID' not in headers:
        _log(logging, 'Port {0} is free.'.format(port))
        return False

    index_pid = headers.index('PID')
    index_cmd = headers.index('COMMAND')
    row = process.stdout.readline().decode().split()
    if len(row) < index_pid:
        _log(logging, 'Port {0} is free.'.format(port))
        return False

    pid = int(row[index_pid])
    command = row[index_cmd]
    
    if pid and command.startswith('python'):
        _log(logging, 'Port {0} is already being used by process {1}!'
             .format(port, pid))
    
        if kill:
            _log(logging,
                 'Killing process with id {0} listening on port {1}!'
                 .format(pid, port))
            os.kill(pid, signal.SIGKILL)

            # Check whether it was really killed.
            try:
                # If still alive
                kill_process(pid, logging)
                # call me again
                _log(logging,
                     'Process {0} is still alive! checking again...'
                     .format(pid))
                return port_in_use(port, kill)
            except OSError:
                # If killed
                return False
        else:
            return pid