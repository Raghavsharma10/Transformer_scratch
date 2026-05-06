def sleep(self, unique_id, delay, configs=None):
    """ Pauses the process for the specified delay and then resumes it

    :Parameter unique_id: the name of the process
    :Parameter delay: delay time in seconds
    """
    self.pause(unique_id, configs)
    time.sleep(delay)
    self.resume(unique_id, configs)