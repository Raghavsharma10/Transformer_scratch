def stop(self, unique_id, configs=None):
    """Stop the service.  If the deployer has not started a service with`unique_id` the deployer will raise an Exception
    There are two configs that will be considered:
    'terminate_only': if this config is passed in then this method is the same as terminate(unique_id) (this is also the
    behavior if stop_command is None and not overridden)
    'stop_command': overrides the default stop_command

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

    logger.debug("stopping " + unique_id)

    if unique_id in self.processes:
      hostname = self.processes[unique_id].hostname
    else:
      logger.error("Can't stop {0}: process not known".format(unique_id))
      raise DeploymentError("Can't stop {0}: process not known".format(unique_id))

    if configs.get('terminate_only', False):
      self.terminate(unique_id, configs)
    else:
      stop_command = configs.get('stop_command') or self.default_configs.get('stop_command')
      env = configs.get("env", {})
      if stop_command is not None:
        install_path = self.processes[unique_id].install_path
        with get_ssh_client(hostname, username=runtime.get_username(), password=runtime.get_password()) as ssh:
          log_output(exec_with_env(ssh, "cd {0}; {1}".format(install_path, stop_command),
                                         msg="Failed to stop {0}".format(unique_id), env=env))
      else:
        self.terminate(unique_id, configs)

    if 'delay' in configs:
      time.sleep(configs['delay'])