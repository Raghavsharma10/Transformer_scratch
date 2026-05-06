def setup(args):
  """Returns the JobManager and sets up the basic infrastructure"""

  kwargs = {'wrapper_script' : args.wrapper_script, 'debug' : args.verbose==3, 'database' : args.database}
  if args.local:
    jm = local.JobManagerLocal(**kwargs)
  else:
    jm = sge.JobManagerSGE(**kwargs)

  # set-up logging
  if args.verbose not in range(0,4):
    raise ValueError("The verbosity level %d does not exist. Please reduce the number of '--verbose' parameters in your call to maximum 3" % level)

  # set up the verbosity level of the logging system
  log_level = {
      0: logging.ERROR,
      1: logging.WARNING,
      2: logging.INFO,
      3: logging.DEBUG
    }[args.verbose]

  handler = logging.StreamHandler()
  handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
  logger.addHandler(handler)
  logger.setLevel(log_level)

  return jm