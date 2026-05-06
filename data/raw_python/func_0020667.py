def resume(self, unique_id, configs=None):
    """ Issues a sigcont for the specified process

    :Parameter unique_id: the name of the process
    """
    self._send_signal(unique_id, signal.SIGCONT,configs)