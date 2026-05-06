def check_for_new(self):
        """Check if we can start a new process."""
        free_slots = self.max_processes - len(self.processes)
        for item in range(free_slots):
            key = self.queue.next()
            if key is not None:
                self.spawn_new(key)