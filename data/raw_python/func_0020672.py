def fetch_logs(self, unique_id, logs, directory, pattern=constants.FILTER_NAME_ALLOW_NONE):
    """ Copies logs from the remote host that the process is running on to the provided directory

    :Parameter unique_id the unique_id of the process in question
    :Parameter logs a list of logs given by absolute path from the remote host
    :Parameter directory the local directory to store the copied logs
    :Parameter pattern a pattern to apply to files to restrict the set of logs copied
    """
    hostname = self.processes[unique_id].hostname
    install_path = self.processes[unique_id].install_path
    self.fetch_logs_from_host(hostname, install_path, unique_id, logs, directory, pattern)