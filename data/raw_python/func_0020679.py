def get_host(self, unique_id):
    """Gets the host of the process with `unique_id`.  If the deployer does not know of a process
    with `unique_id` then it should return a value of SOME_SENTINAL_VALUE

    :Parameter unique_id: the name of the process
    :raises NameError if the name is not  valid process
    """
    if unique_id in self.processes:
      return self.processes[unique_id].hostname
    logger.error("{0} not a known process".format(unique_id))
    raise NameError("{0} not a known process".format(unique_id))