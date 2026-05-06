def format(self, format, dependencies = 0, limit_command_line = None):
    """Formats the current job into a nicer string to fit into a table."""
    command_line = self._cmdline()
    if limit_command_line is not None and len(command_line) > limit_command_line:
      command_line = command_line[:limit_command_line-3] + '...'

    job_id = "%d" % self.id + (" [%d-%d:%d]" % self.get_array() if self.array else "")
    status = "%s" % self.status + (" (%d)" % self.result if self.result is not None else "" )
    queue = self.queue_name if self.machine_name is None else self.machine_name
    if limit_command_line is None:
      grid_opt = self.get_arguments()
      if grid_opt:
        # add additional information about the job at the end
        command_line = "<" + ",".join(["%s=%s" % (key,value) for key,value in grid_opt.items()]) + ">: " + command_line
      if self.exec_dir is not None:
        command_line += "; [Executed in directory: '%s']" % self.exec_dir

    if dependencies:
      deps = str(sorted(list(set([dep.unique for dep in self.get_jobs_we_wait_for()]))))
      if dependencies < len(deps):
        deps = deps[:dependencies-3] + '...'
      return format.format(self.unique, job_id, queue[:12], status, self.name, deps, command_line)
    else:
      return format.format(self.unique, job_id, queue[:12], status, self.name, command_line)