def uninstall(self, unique_id, configs=None):
    """uninstall the service.  If the deployer has not started a service with
    `unique_id` this will raise a DeploymentError.  This considers one config:
    'additional_directories': a list of directories to remove in addition to those provided in the constructor plus
     the install path. This will update the directories to remove but does not override it
    :param unique_id:
    :param configs:
    :return:
    """
    # the following is necessay to set the configs for this function as the combination of the
    # default configurations and the parameter with the parameter superceding the defaults but
    # not modifying the defaults
    if configs is None:
      configs = {}
    tmp = self.default_configs.copy()
    tmp.update(configs)
    configs = tmp

    if unique_id in self.processes:
      hostname = self.processes[unique_id].hostname
    else:
      logger.error("Can't uninstall {0}: process not known".format(unique_id))
      raise DeploymentError("Can't uninstall {0}: process not known".format(unique_id))

    install_path = self.processes[unique_id].install_path
    directories_to_remove = self.default_configs.get('directories_to_clean', [])
    directories_to_remove.extend(configs.get('additional_directories', []))
    if install_path not in directories_to_remove:
      directories_to_remove.append(install_path)
    with get_ssh_client(hostname, username=runtime.get_username(), password=runtime.get_password()) as ssh:
      for directory_to_remove in directories_to_remove:
        log_output(better_exec_command(ssh, "rm -rf {0}".format(directory_to_remove),
                                       "Failed to remove {0}".format(directory_to_remove)))