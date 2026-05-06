def submit(jman, command, arguments, deps=[], array=None):
  """An easy submission option for grid-enabled scripts. Create the log
  directories using random hash codes. Use the arguments as parsed by the main
  script."""

  logdir = os.path.join(os.path.realpath(arguments.logdir),
      tools.random_logdir())

  jobname = os.path.splitext(os.path.basename(command[0]))[0]
  cmd = tools.make_shell(sys.executable, command)

  if arguments.dryrun:
    return DryRunJob(cmd, cwd=arguments.cwd, queue=arguments.queue,
        hostname=arguments.hostname, memfree=arguments.memfree,
        hvmem=arguments.hvmem, gpumem=arguments.gpumem, pe_opt=arguments.pe_opt,
        stdout=logdir, stderr=logdir, name=jobname, deps=deps,
        array=array)
  
  # really submit
  return jman.submit(cmd, cwd=arguments.cwd, queue=arguments.queue,
      hostname=arguments.hostname, memfree=arguments.memfree,
      hvmem=arguments.hvmem, gpumem=arguments.gpumem, pe_opt=arguments.pe_opt,
      stdout=logdir, stderr=logdir, name=jobname, deps=deps,
      array=array)