def run_scheduler(self, parallel_jobs = 1, job_ids = None, sleep_time = 0.1, die_when_finished = False, no_log = False, nice = None, verbosity = 0):
    """Starts the scheduler, which is constantly checking for jobs that should be ran."""
    running_tasks = []
    finished_tasks = set()
    try:

      # keep the scheduler alive until every job is finished or the KeyboardInterrupt is caught
      while True:
        # Flag that might be set in some rare cases, and that prevents the scheduler to die
        repeat_execution = False
        # FIRST, try if there are finished processes
        for task_index in range(len(running_tasks)-1, -1, -1):
          task = running_tasks[task_index]
          process = task[0]

          if process.poll() is not None:
            # process ended
            job_id = task[1]
            array_id = task[2] if len(task) > 2 else None
            self.lock()
            job, array_job = self._job_and_array(job_id, array_id)
            if job is not None:
              jj = array_job if array_job is not None else job
              result = "%s (%d)" % (jj.status, jj.result) if jj.result is not None else "%s (?)" % jj.status
              if jj.status not in ('success', 'failure'):
                logger.error("Job '%s' (%s) finished with status '%s' instead of 'success' or 'failure'. Usually this means an internal error. Check your wrapper_script parameter!", job.name, self._format_log(job_id, array_id), jj.status)
                raise StopIteration("Job did not finish correctly.")
              logger.info("Job '%s' (%s) finished execution with result '%s'", job.name, self._format_log(job_id, array_id), result)
            self.unlock()
            finished_tasks.add(job_id)
            # in any case, remove the job from the list
            del running_tasks[task_index]

        # SECOND, check if new jobs can be submitted; THIS NEEDS TO LOCK THE DATABASE
        if len(running_tasks) < parallel_jobs:
          # get all unfinished jobs:
          self.lock()
          jobs = self.get_jobs(job_ids)
          # put all new jobs into the queue
          for job in jobs:
            if job.status == 'submitted' and job.queue_name == 'local':
              job.queue()

          # get all unfinished jobs that are submitted to the local queue
          unfinished_jobs = [job for job in jobs if job.status in ('queued', 'executing') and job.queue_name == 'local']
          for job in unfinished_jobs:
            if job.array:
              # find array jobs that can run
              queued_array_jobs = [array_job for array_job in job.array if array_job.status == 'queued']
              if not len(queued_array_jobs):
                job.finish(0, -1)
                repeat_execution = True
              else:
                # there are new array jobs to run
                for i in range(min(parallel_jobs - len(running_tasks), len(queued_array_jobs))):
                  array_job = queued_array_jobs[i]
                  # start a new job from the array
                  process = self._run_parallel_job(job.unique, array_job.id, no_log=no_log, nice=nice, verbosity=verbosity)
                  if process is None:
                    continue
                  running_tasks.append((process, job.unique, array_job.id))
                  # we here set the status to executing manually to avoid jobs to be run twice
                  # e.g., if the loop is executed while the asynchronous job did not start yet
                  array_job.status = 'executing'
                  job.status = 'executing'
                  if len(running_tasks) == parallel_jobs:
                    break
            else:
              if job.status == 'queued':
                # start a new job
                process = self._run_parallel_job(job.unique, no_log=no_log, nice=nice, verbosity=verbosity)
                if process is None:
                  continue
                running_tasks.append((process, job.unique))
                # we here set the status to executing manually to avoid jobs to be run twice
                # e.g., if the loop is executed while the asynchronous job did not start yet
                job.status = 'executing'
            if len(running_tasks) == parallel_jobs:
              break

          self.session.commit()
          self.unlock()

        # if after the submission of jobs there are no jobs running, we should have finished all the queue.
        if die_when_finished and not repeat_execution and len(running_tasks) == 0:
          logger.info("Stopping task scheduler since there are no more jobs running.")
          break

        # THIRD: sleep the desired amount of time before re-checking
        time.sleep(sleep_time)

    # This is the only way to stop: you have to interrupt the scheduler
    except (KeyboardInterrupt, StopIteration):
      if hasattr(self, 'session'):
        self.unlock()
      logger.info("Stopping task scheduler due to user interrupt.")
      for task in running_tasks:
        logger.warn("Killing job '%s' that was still running.", self._format_log(task[1], task[2] if len(task) > 2 else None))
        try:
          task[0].kill()
        except OSError as e:
          logger.error("Killing job '%s' was not successful: '%s'", self._format_log(task[1], task[2] if len(task) > 2 else None), e)
        self.stop_job(task[1])
      # stop all jobs that are currently running or queued
      self.stop_jobs(job_ids)

    # check the result of the jobs that we have run, and return the list of failed jobs
    self.lock()
    jobs = self.get_jobs(finished_tasks)
    failures = [job.unique for job in jobs if job.status != 'success']
    self.unlock()
    return sorted(failures)