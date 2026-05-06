def remove_go(self, target):
        """
        FOR SAVING MEMORY
        """
        with self.lock:
            if not self._go:
                try:
                    self.job_queue.remove(target)
                except ValueError:
                    pass