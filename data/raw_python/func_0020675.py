def start(self, unique_id, configs=None):
    """
    Start the service.  If `unique_id` has already been installed the deployer will start the service on that host.
    Otherwise this will call install with the configs. Within the context of this function, only four configs are
    considered
    'start_command': the command to run (if provided will replace the default)
    'args': a list of args that can be passed to the command
    'sync': if the command is synchronous or asynchronous defaults to asynchronous
    'delay': a delay in seconds that might be needed regardless of whether the command returns before the service can
    be started

    :param unique_id:
    :param configs:
    :return: if the command is executed synchronously return the underlying paramiko channel which can be used to get the stdout
    otherwise return the triple stdin, stdout, stderr
    """
    # the following is necessay to set the configs for this function as the combination of the
    # default configurations and the parameter with the parameter superceding the defaults but
    # not modifying the defaults
    if configs is None:
      configs = {}
    tmp = self.default_configs.copy()
    tmp.update(configs)
    configs = tmp

    logger.debug("starting " + unique_id)

    # do not start if already started
    if self.get_pid(unique_id, configs) is not constants.PROCESS_NOT_RUNNING_PID:
      return None

    if unique_id not in self.processes:
      self.install(unique_id, configs)

    hostname = self.processes[unique_id].hostname
    install_path = self.processes[unique_id].install_path

    # order of precedence for start_command and args from highest to lowest:
    # 1. configs
    # 2. from Process
    # 3. from Deployer
    start_command = configs.get('start_command') or self.processes[unique_id].start_command or self.default_configs.get('start_command')
    pid_file = configs.get('pid_file') or self.default_configs.get('pid_file')
    if start_command is None:
      logger.error("start_command was not provided for unique_id: " + unique_id)
      raise DeploymentError("start_command was not provided for unique_id: " + unique_id)
    args = configs.get('args') or self.processes[unique_id].args or self.default_configs.get('args')
    if args is not None:
      full_start_command = "{0} {1}".format(start_command, ' '.join(args))
    else:
      full_start_command = start_command
    command = "cd {0}; {1}".format(install_path, full_start_command)
    env = configs.get("env", {})
    with get_ssh_client(hostname, username=runtime.get_username(), password=runtime.get_password()) as ssh:
      exec_with_env(ssh, command, msg="Failed to start", env=env, sync=configs.get('sync', False))

    self.processes[unique_id].start_command = start_command
    self.processes[unique_id].args = args
    # For cases where user pases it with start command
    if self.processes[unique_id].pid_file is None:
      self.processes[unique_id].pid_file = pid_file

    if 'delay' in configs:
      time.sleep(configs['delay'])