def run(self):
        """ Listen to the stream and send events to the client. """
        channel = self._ssh_client.get_transport().open_session()
        self._channel = channel
        channel.exec_command("gerrit stream-events")
        stdout = channel.makefile()
        stderr = channel.makefile_stderr()
        while not self._stop.is_set():
            try:
                if channel.exit_status_ready():
                    if channel.recv_stderr_ready():
                        error = stderr.readline().strip()
                    else:
                        error = "Remote server connection closed"
                    self._error_event(error)
                    self._stop.set()
                else:
                    data = stdout.readline()
                    self._gerrit.put_event(data)
            except Exception as e:  # pylint: disable=W0703
                self._error_event(repr(e))
                self._stop.set()