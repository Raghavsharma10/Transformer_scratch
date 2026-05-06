def communicate(args):
  """Uses qstat to get the status of the requested jobs."""
  if args.local:
    raise ValueError("The communicate command can only be used without the '--local' command line option")
  jm = setup(args)
  jm.communicate(job_ids=get_ids(args.job_ids))