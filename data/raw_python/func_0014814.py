def _stream_docker_logs(self):
        """Stream stdout and stderr from the task container to this
        process's stdout and stderr, respectively.
        """
        thread = threading.Thread(target=self._stderr_stream_worker)
        thread.start()
        for line in self.docker_client.logs(self.container, stdout=True,
                                            stderr=False, stream=True):
            sys.stdout.write(line)
        thread.join()