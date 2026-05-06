def kill_all_process(self):
    """ Terminates all the running processes. By default it is set to false.
    Users can set to true in config once the method to get_pid is done deterministically
    either using pid_file or an accurate keyword

    """
    if (runtime.get_active_config("cleanup_pending_process",False)):
      for process in self.get_processes():
        self.terminate(process.unique_id)