def shutdown(self):
        """Shutdown the accept loop and stop running payloads"""
        self._must_shutdown = True
        self._is_shutdown.wait()
        self._meta_runner.stop()