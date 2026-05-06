def new_worker_pool(self, name: str, min_workers: int = 0, max_workers: int = 1,
                        max_seconds_idle: int = DEFAULT_WORKER_POOL_MAX_SECONDS_IDLE):
        """
        Creates a new worker pool and starts it.
        Returns the Worker that schedules works to the pool.
        """
        if not self.running:
            return self.immediate_worker
        worker = self._new_worker_pool(name, min_workers, max_workers, max_seconds_idle)
        self._start_worker_pool(worker)
        return worker