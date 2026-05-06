def shell_exec(command, **kwargs): # from gitapi.py
  """Excecutes the given command silently.
  """
  proc = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE, **kwargs)

  out, err = [x.decode("utf-8") for x in  proc.communicate()]

  return {'out': out, 'err': err, 'code': proc.returncode}