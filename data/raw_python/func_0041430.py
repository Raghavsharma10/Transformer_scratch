def create_manager(arguments):
  """A simple wrapper to JobManager() that places the statefile on the correct path by default"""

  if arguments.statefile is None:
    arguments.statefile = os.path.join(os.path.dirname(arguments.logdir), 'submitted.db')

  arguments.statefile = os.path.realpath(arguments.statefile)

  return manager.JobManager(statefile=arguments.statefile)