def report(args):
  """Reports the results of the finished (and unfinished) jobs."""
  jm = setup(args)
  jm.report(job_ids=get_ids(args.job_ids), array_ids=get_ids(args.array_ids), output=not args.errors_only, error=not args.output_only, status=args.status, name=args.name)