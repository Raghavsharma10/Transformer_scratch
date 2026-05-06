def _kill_worker_threads(self):
        """This function coerces the consumer/worker threads to kill
        themselves.  When called by the queuing thread, one death token will
        be placed on the queue for each thread.  Each worker thread is always
        looking for the death token.  When it encounters it, it immediately
        runs to completion without drawing anything more off the queue.

        This is a blocking call.  The thread using this function will wait for
        all the worker threads to die."""
        for x in range(self.number_of_threads):
            self.task_queue.put((None, None))
        self.logger.debug("waiting for standard worker threads to stop")
        for t in self.thread_list:
            t.join()