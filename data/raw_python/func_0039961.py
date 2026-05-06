def list(args):
  """Lists the jobs in the given database."""
  jm = setup(args)
  jm.list(job_ids=get_ids(args.job_ids), print_array_jobs=args.print_array_jobs, print_dependencies=args.print_dependencies, status=args.status, long=args.long, print_times=args.print_times, ids_only=args.ids_only, names=args.names)