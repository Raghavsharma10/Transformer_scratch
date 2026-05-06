def hangup(self, unique_id, configs=None):
    """
    Issue a signal to hangup the specified process

    :Parameter unique_id: the name of the process
    """
    self._send_signal(unique_id, signal.SIGHUP, configs)