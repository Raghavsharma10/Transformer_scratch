def run_job(args):
  """Starts the wrapper script to execute a job, interpreting the JOB_ID and SGE_TASK_ID keywords that are set by the grid or by us."""
  jm = setup(args)
  job_id = int(os.environ['JOB_ID'])
  array_id = int(os.environ['SGE_TASK_ID']) if os.environ['SGE_TASK_ID'] != 'undefined' else None
  jm.run_job(job_id, array_id)