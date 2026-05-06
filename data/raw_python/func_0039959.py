def resubmit(args):
  """Re-submits the jobs with the given ids."""
  jm = setup(args)

  kwargs = {
      'cwd': True,
      'verbosity' : args.verbose
  }
  if args.qname is not None:
    kwargs['queue'] = args.qname
  if args.memory is not None:
    kwargs['memfree'] = args.memory
    if args.qname not in (None, 'all.q'):
      kwargs['hvmem'] = args.memory
    # if this is a GPU queue and args.memory is provided, we set gpumem flag
    # remove 'G' last character from the args.memory string
    if args.qname in ('gpu', 'lgpu', 'sgpu', 'gpum') and args.memory is not None:
      kwargs['gpumem'] = args.memory
      # don't set these for GPU processing or the maximum virtual memroy will be
      # set on ulimit
      kwargs.pop('memfree', None)
      kwargs.pop('hvmem', None)
  if args.parallel is not None:
    kwargs['pe_opt'] = "pe_mth %d" % args.parallel
    kwargs['memfree'] = get_memfree(args.memory, args.parallel)
  if args.io_big:
    kwargs['io_big'] = True
  if args.no_io_big:
    kwargs['io_big'] = False

  jm.resubmit(get_ids(args.job_ids), args.also_success, args.running_jobs, args.overwrite_command, keep_logs=args.keep_logs, **kwargs)