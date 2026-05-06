def soft_bounce(self, unique_id, configs=None):
    """ Performs a soft bounce (stop and start) for the specified process

    :Parameter unique_id: the name of the process
    """
    self.stop(unique_id, configs)
    self.start(unique_id, configs)