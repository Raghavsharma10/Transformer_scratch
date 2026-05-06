def pause(self, unique_id, configs=None):
    """ Issues a sigstop for the specified process

    :Parameter unique_id: the name of the process
    """
    pids = self.get_pid(unique_id, configs)
    if pids != constants.PROCESS_NOT_RUNNING_PID:
      pid_str = ' '.join(str(pid) for pid in pids)
      hostname = self.processes[unique_id].hostname
      with get_ssh_client(hostname, username=runtime.get_username(), password=runtime.get_password()) as ssh:
        better_exec_command(ssh, "kill -SIGSTOP {0}".format(pid_str), "PAUSING PROCESS {0}".format(unique_id))