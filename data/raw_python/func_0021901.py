def new_worker(self, name: str):
        """Creates a new Worker and start a new Thread with it. Returns the Worker."""
        if not self.running:
            return self.immediate_worker
        worker = self._new_worker(name)
        self._start_worker(worker)
        return worker