def _run_parallel_job(self, job_id, array_id = None, no_log = False, nice = None, verbosity = 0):
    """Executes the code for this job on the local machine."""
    environ = copy.deepcopy(os.environ)
    environ['JOB_ID'] = str(job_id)
    if array_id:
      environ['SGE_TASK_ID'] = str(array_id)
    else:
      environ['SGE_TASK_ID'] = 'undefined'

    # generate call to the wrapper script
    command = [self.wrapper_script, '-l%sd'%("v"*verbosity), self._database, 'run-job']

    if nice is not None:
      command = ['nice', '-n%d'%nice] + command

    job, array_job = self._job_and_array(job_id, array_id)
    if job is None:
      # rare case: job was deleted before starting
      return None

    logger.info("Starting execution of Job '%s' (%s)", job.name, self._format_log(job_id, array_id, len(job.array)))
    # create log files
    if no_log or job.log_dir is None:
      out, err = sys.stdout, sys.stderr
    else:
      makedirs_safe(job.log_dir)
      # create line-buffered files for writing output and error status
      if array_job is not None:
        out, err = open(array_job.std_out_file(), 'w', 1), open(array_job.std_err_file(), 'w', 1)
      else:
        out, err = open(job.std_out_file(), 'w', 1), open(job.std_err_file(), 'w', 1)

    # return the subprocess pipe to the process
    try:
      return subprocess.Popen(command, env=environ, stdout=out, stderr=err, bufsize=1)
    except OSError as e:
      logger.error("Could not execute job '%s' (%s) locally\n- reason:\t%s\n- command line:\t%s\n- directory:\t%s\n- command:\t%s", job.name, self._format_log(job_id, array_id, len(job.array)), e, " ".join(job.get_command_line()), "." if job.exec_dir is None else job.exec_dir, " ".join(command))
      job.finish(117, array_id) # ASCII 'O'
      return None