def run_scheduler(args):
  """Runs the scheduler on the local machine. To stop it, please use Ctrl-C."""
  if not args.local:
    raise ValueError("The execute command can only be used with the '--local' command line option")
  jm = setup(args)
  jm.run_scheduler(parallel_jobs=args.parallel, job_ids=get_ids(args.job_ids), sleep_time=args.sleep_time, die_when_finished=args.die_when_finished, no_log=args.no_log_files, nice=args.nice, verbosity=args.verbose)