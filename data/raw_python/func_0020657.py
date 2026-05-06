def better_exec_command(ssh, command, msg):
  """Uses paramiko to execute a command but handles failure by raising a ParamikoError if the command fails.
  Note that unlike paramiko.SSHClient.exec_command this is not asynchronous because we wait until the exit status is known

  :Parameter ssh: a paramiko SSH Client
  :Parameter command: the command to execute
  :Parameter msg: message to print on failure

  :Returns (paramiko.Channel)
   the underlying channel so that the caller can extract stdout or send to stdin

  :Raises  SSHException: if paramiko would raise an SSHException
  :Raises  ParamikoError: if the command produces output to stderr
  """
  chan = ssh.get_transport().open_session()
  chan.exec_command(command)
  exit_status = chan.recv_exit_status()
  if exit_status != 0:
    msg_str = chan.recv_stderr(1024)
    err_msgs = []
    while len(msg_str) > 0:
      err_msgs.append(msg_str)
      msg_str = chan.recv_stderr(1024)
    err_msg = ''.join(err_msgs)
    logger.error(err_msg)
    raise ParamikoError(msg, err_msg)
  return chan