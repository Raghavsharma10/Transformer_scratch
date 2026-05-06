def makedirs_safe(fulldir):
  """Creates a directory if it does not exists. Takes into consideration
  concurrent access support. Works like the shell's 'mkdir -p'.
  """

  try:
    if not os.path.exists(fulldir): os.makedirs(fulldir)
  except OSError as exc: # Python >2.5
    import errno
    if exc.errno == errno.EEXIST: pass
    else: raise