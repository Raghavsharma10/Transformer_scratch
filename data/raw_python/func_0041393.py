def environ(context):
  """Retrieves the environment for a particular SETSHELL context"""
  if 'BASEDIRSETSHELL' not in os.environ:
    # It seems that we are in a hostile environment
    # try to source the Idiap-wide shell
    idiap_source = "/idiap/resource/software/initfiles/shrc"
    if os.path.exists(idiap_source):
      logger.debug("Sourcing: '%s'"%idiap_source)
      try:
        command = ['bash', '-c', 'source %s && env' % idiap_source]
        pi = subprocess.Popen(command, stdout = subprocess.PIPE)
        # overwrite the default environment
        for line in pi.stdout:
          line = str_(line)
          (key, _, value) = line.partition("=")
          os.environ[key.strip()] = value.strip()
      except OSError as e:
        # occurs when the file is not executable or not found
        pass

  # in case the BASEDIRSETSHELL environment variable is not set,
  # we are not at Idiap,
  # and so we don't have to set any additional variables.
  if 'BASEDIRSETSHELL' not in os.environ:
    return dict(os.environ)

  BASEDIRSETSHELL = os.environ['BASEDIRSETSHELL']
  dosetshell = '%s/setshell/bin/dosetshell' % BASEDIRSETSHELL

  command = [dosetshell, '-s', 'sh', context]

  # First things first, we get the path to the temp file created by dosetshell
  try:
    logger.debug("Executing: '%s'", ' '.join(command))
    p = subprocess.Popen(command, stdout = subprocess.PIPE)
  except OSError as e:
    # occurs when the file is not executable or not found
    raise OSError("Error executing '%s': %s (%d)" % (' '.join(command), e.strerror, e.errno))

  try:
    source = str_(p.communicate()[0]).strip()
  except KeyboardInterrupt: # the user CTRL-C'ed
    os.kill(p.pid, signal.SIGTERM)
    sys.exit(signal.SIGTERM)

  # We have now the name of the source file, source it and erase it
  command2 = ['bash', '-c', 'source %s && env' % source]

  try:
    logger.debug("Executing: '%s'", ' '.join(command2))
    p2 = subprocess.Popen(command2, stdout = subprocess.PIPE)
  except OSError as e:
    # occurs when the file is not executable or not found
    raise OSError("Error executing '%s': %s (%d)" % (' '.join(command2), e.strerror, e.errno))

  new_environ = dict(os.environ)
  for line in p2.stdout:
    line = str_(line)
    (key, _, value) = line.partition("=")
    new_environ[key.strip()] = value.strip()

  try:
    p2.communicate()
  except KeyboardInterrupt: # the user CTRL-C'ed
    os.kill(p2.pid, signal.SIGTERM)
    sys.exit(signal.SIGTERM)

  if os.path.exists(source): os.unlink(source)

  logger.debug("Discovered environment for context '%s':", context)
  for k in sorted(new_environ.keys()):
    logger.debug("  %s = %s", k, new_environ[k])

  return new_environ