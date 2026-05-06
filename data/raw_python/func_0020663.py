def hard_bounce(self, unique_id, configs=None):
    """ Performs a hard bounce (kill and start) for the specified process

    :Parameter unique_id: the name of the process
    """
    self.kill(unique_id, configs)
    self.start(unique_id, configs)