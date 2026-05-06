def submit(args):
  """Submission command"""

  # set full path to command
  if args.job[0] == '--':
    del args.job[0]
  if not os.path.isabs(args.job[0]):
    args.job[0] = os.path.abspath(args.job[0])

  jm = setup(args)
  kwargs = {
      'queue': args.qname,
      'cwd': True,
      'verbosity' : args.verbose,
      'name': args.name,
      'env': args.env,
      'memfree': args.memory,
      'io_big': args.io_big,
  }

  if args.array is not None:         kwargs['array'] = get_array(args.array)
  if args.exec_dir is not None:      kwargs['exec_dir'] = args.exec_dir
  if args.log_dir is not None:       kwargs['log_dir'] = args.log_dir
  if args.dependencies is not None:  kwargs['dependencies'] = args.dependencies
  if args.qname != 'all.q':          kwargs['hvmem'] = args.memory
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
    if args.memory is not None:
      kwargs['memfree'] = get_memfree(args.memory, args.parallel)
  kwargs['dry_run'] = args.dry_run
  kwargs['stop_on_failure'] = args.stop_on_failure

  # submit the job(s)
  for _ in range(args.repeat):
    job_id = jm.submit(args.job, **kwargs)
    dependencies = kwargs.get('dependencies', [])
    dependencies.append(job_id)
    kwargs['dependencies'] = dependencies

  if args.print_id:
    print (job_id, end='')