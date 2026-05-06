def open_remote_file(hostname, filename, mode='r', bufsize=-1, username=None, password=None):
  """

  :param hostname:
  :param filename:
  :return:
  """
  with get_ssh_client(hostname, username=username, password=password) as ssh:
    sftp = None
    f = None
    try:
      sftp = ssh.open_sftp()
      f = sftp.open(filename, mode, bufsize)
      yield f
    finally:
      if f is not None:
        f.close()
      if sftp is not None:
        sftp.close()