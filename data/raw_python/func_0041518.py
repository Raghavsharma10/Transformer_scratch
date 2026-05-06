def run_job(self, job_id, array_id = None):
    """This function is called to run a job (e.g. in the grid) with the given id and the given array index if applicable."""
    # set the job's status in the database
    try:
      # get the job from the database
      self.lock()
      jobs = self.get_jobs((job_id,))
      if not len(jobs):
        # it seems that the job has been deleted in the meanwhile
        return
      job = jobs[0]

      # get the machine name we are executing on; this might only work at idiap
      machine_name = socket.gethostname()

      # set the 'executing' status to the job
      job.execute(array_id, machine_name)

      self.session.commit()
    except Exception as e:
      logger.error("Caught exception '%s'", e)
      pass
    finally:
      self.unlock()

    # get the command line of the job from the database; does not need write access
    self.lock()
    job = self.get_jobs((job_id,))[0]
    command_line = job.get_command_line()
    exec_dir = job.get_exec_dir()
    self.unlock()

    logger.info("Starting job %d: %s", job_id, " ".join(command_line))

    # execute the command line of the job, and wait until it has finished
    try:
      result = subprocess.call(command_line, cwd=exec_dir)
      logger.info("Job %d finished with result %s", job_id, str(result))
    except Exception as e:
      logger.error("The job with id '%d' could not be executed: %s", job_id, e)
      result = 69 # ASCII: 'E'

    # set a new status and the results of the job
    try:
      self.lock()
      jobs = self.get_jobs((job_id,))
      if not len(jobs):
        # it seems that the job has been deleted in the meanwhile
        logger.error("The job with id '%d' could not be found in the database!", job_id)
        self.unlock()
        return

      job = jobs[0]
      job.finish(result, array_id)

      self.session.commit()

      # This might not be working properly, so use with care!
      if job.stop_on_failure and job.status == 'failure':
        # the job has failed
        # stop this and all dependent jobs from execution
        dependent_jobs = job.get_jobs_waiting_for_us()
        dependent_job_ids = set([dep.unique for dep in dependent_jobs] + [job.unique])
        while len(dependent_jobs):
          dep = dependent_jobs.pop(0)
          new = dep.get_jobs_waiting_for_us()
          dependent_jobs += new
          dependent_job_ids.update([dep.unique for dep in new])

        self.unlock()
        deps = sorted(list(dependent_job_ids))
        self.stop_jobs(deps)
        logger.warn ("Stopped dependent jobs '%s' since this job failed.", str(deps))

    except Exception as e:
      logger.error("Caught exception '%s'", e)
      pass
    finally:
      if hasattr(self, 'session'):
        self.unlock()