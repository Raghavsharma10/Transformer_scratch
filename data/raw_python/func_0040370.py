def handler_thread(self):
        """
        A handler thread--this pulls results from the queue and processes them
        accordingly.

        Calls a sub method based on type of monitor.
        """
        self.thread_debug("Starting handler thread")
        while not self.thread_stopper.is_set():
            data = self.results_queue.get()
            self.thread_debug("Handling Result", module="handler")
            getattr(self, "_handler_" + data['type'])(data['result'])