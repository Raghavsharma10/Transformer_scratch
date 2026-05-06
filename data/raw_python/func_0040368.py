def worker_thread(self):
        """
        The primary worker thread--this thread pulls from the monitor queue and
        runs the monitor, submitting the results to the handler queue.

        Calls a sub method based on type of monitor.
        """
        self.thread_debug("Starting monitor thread")
        while not self.thread_stopper.is_set():
            mon = self.workers_queue.get()
            self.thread_debug("Processing {type} Monitor: {title}".format(**mon))
            result = getattr(self, "_worker_" + mon['type'])(mon)
            self.workers_queue.task_done()
            self.results_queue.put({'type':mon['type'], 'result':result})