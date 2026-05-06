def deploy(self, unique_id, configs=None):
    """Deploys the service to the host.  This should at least perform the same actions as install and start
    but may perform additional tasks as needed.

    :Parameter unique_id: the name of the process
    :Parameter configs: a mao of configs the deployer may use to modify the deployment
    """
    self.install(unique_id, configs)
    self.start(unique_id, configs)