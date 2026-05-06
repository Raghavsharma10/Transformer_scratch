def terminate(self, unique_id, configs=None):
    """ Issues a kill -15 to the specified process

    :Parameter unique_id: the name of the process
    """
    self._send_signal(unique_id, signal.SIGTERM, configs)