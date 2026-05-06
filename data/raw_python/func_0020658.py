def log_output(chan):
  """
  logs the output from a remote command
  the input should be an open channel in  the case of synchronous better_exec_command
  otherwise this will not log anything and simply return to the caller
  :param chan:
  :return:
  """
  if hasattr(chan, "recv"):
    str = chan.recv(1024)
    msgs = []
    while len(str) > 0:
      msgs.append(str)
      str = chan.recv(1024)
    msg = ''.join(msgs).strip()
    if len(msg) > 0:
      logger.info(msg)