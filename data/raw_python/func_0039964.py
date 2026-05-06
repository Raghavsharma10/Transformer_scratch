def stop(args):
  """Stops (qdel's) the jobs with the given ids."""
  if args.local:
    raise ValueError("Stopping commands locally is not supported (please kill them yourself)")
  jm = setup(args)
  jm.stop_jobs(get_ids(args.job_ids))