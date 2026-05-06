def list(self, job_ids, print_array_jobs = False, print_dependencies = False, long = False, print_times = False, status=Status, names=None, ids_only=False):
    """Lists the jobs currently added to the database."""
    # configuration for jobs
    fields = ("job-id", "grid-id", "queue", "status", "job-name")
    lengths = (6, 17, 11, 12, 16)
    dependency_length = 0

    if print_dependencies:
      fields += ("dependencies",)
      lengths += (25,)
      dependency_length = lengths[-1]

    if long:
      fields += ("submitted command",)
      lengths += (43,)

    format = "{:^%d}  " * len(lengths)
    format = format % lengths

    # if ids_only:
    #   self.lock()
    #   for job in self.get_jobs():
    #     print(job.unique, end=" ")
    #   self.unlock()
    #   return

    array_format = "{0:^%d}  {1:>%d}  {2:^%d}  {3:^%d}" % lengths[:4]
    delimiter = format.format(*['='*k for k in lengths])
    array_delimiter = array_format.format(*["-"*k for k in lengths[:4]])
    header = [fields[k].center(lengths[k]) for k in range(len(lengths))]

    # print header
    if not ids_only:
      print('  '.join(header))
      print(delimiter)

    self.lock()
    for job in self.get_jobs(job_ids):
      job.refresh()
      if job.status in status and (names is None or job.name in names):
        if ids_only:
          print(job.unique, end=" ")
        else:
          print(job.format(format, dependency_length))
        if print_times:
          print(times(job))

        if (not ids_only) and print_array_jobs and job.array:
          print(array_delimiter)
          for array_job in job.array:
            if array_job.status in status:
              print(array_job.format(array_format))
              if print_times:
                print(times(array_job))
          print(array_delimiter)

    self.unlock()