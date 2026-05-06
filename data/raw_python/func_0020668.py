def kill(self, unique_id, configs=None):
    """ Issues a kill -9 to the specified process
    calls the deployers get_pid function for the process. If no pid_file/pid_keyword is specified
    a generic grep of ps aux command is executed on remote machine based on process parameters
    which may not be reliable if more process are running with similar name

    :Parameter unique_id: the name of the process
    """
    self._send_signal(unique_id, signal.SIGKILL, configs)