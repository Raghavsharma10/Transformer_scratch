def delete(args):
  """Deletes the jobs from the job manager. If the jobs are still running in the grid, they are stopped."""
  jm = setup(args)
  # first, stop the jobs if they are running in the grid
  if not args.local and 'executing' in args.status:
    stop(args)
  # then, delete them from the database
  jm.delete(job_ids=get_ids(args.job_ids), array_ids=get_ids(args.array_ids), delete_logs=not args.keep_logs, delete_log_dir=not args.keep_log_dir, status=args.status)