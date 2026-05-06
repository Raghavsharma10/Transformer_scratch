def touch_log(log, cwd='.'):
  """
  Touches the log file. Creates if not exists OR updates the modification date if exists.
  :param log:
  :return: nothing
  """
  logfile = '%s/%s' % (cwd, log)
  with open(logfile, 'a'):
    os.utime(logfile, None)