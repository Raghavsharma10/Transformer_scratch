def undeploy(self, unique_id, configs=None):
    """Undeploys the service.  This should at least perform the same actions as stop and uninstall
    but may perform additional tasks as needed.

    :Parameter unique_id: the name of the process
    :Parameter configs: a map of configs the deployer may use
    """
    self.stop(unique_id, configs)
    self.uninstall(unique_id, configs)