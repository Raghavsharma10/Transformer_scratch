def _batch_entry(self):
        """Entry point for the batcher thread."""
        try:
            while True:
                self._batch_entry_run()
        except:
            self.exc_info = sys.exc_info()
            os.kill(self.pid, signal.SIGUSR1)