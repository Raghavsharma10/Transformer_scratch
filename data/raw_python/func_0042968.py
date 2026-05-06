def kill(self):
        "Kill the daemon instance."
        if self.pid:
            try:
                os.kill(self.pid, signal.SIGTERM)
                # Raises an OSError for ESRCH when we've killed it.
                while True:
                    os.kill(self.pid, signal.SIGTERM)
                    time.sleep(0.01)
            except OSError:
                pass
            self.pid = None