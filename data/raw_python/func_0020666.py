def _send_signal(self, unique_id, signalno, configs):
    """ Issues a signal for the specified process

    :Parameter unique_id: the name of the process
    """
    pids = self.get_pid(unique_id, configs)
    if pids != constants.PROCESS_NOT_RUNNING_PID:
      pid_str = ' '.join(str(pid) for pid in pids)
      hostname = self.processes[unique_id].hostname
      msg=  Deployer._signalnames.get(signalno,"SENDING SIGNAL %s TO"%signalno)
      with get_ssh_client(hostname, username=runtime.get_username(), password=runtime.get_password()) as ssh:
        better_exec_command(ssh, "kill -{0} {1}".format(signalno, pid_str), "{0} PROCESS {1}".format(msg, unique_id))