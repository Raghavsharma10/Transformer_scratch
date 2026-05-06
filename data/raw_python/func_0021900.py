def _start_worker(self, worker: Worker):
        """
        Can be safely called multiple times on the same worker (for workers that support it)
        to start a new thread for it.
        """
        # This function is called from main thread and from worker pools threads to start their children threads
        with self.running_workers_lock:
            self.running_workers.append(worker)
        thread = SchedulerThread(worker, self._worker_ended)
        thread.start()
        # This may or may not be posted to a background thread (see set_callbacks)
        self.worker_start_callback(worker)