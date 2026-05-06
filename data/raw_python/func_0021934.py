def _get_and_execute(self):
        """
        :return: True if it should continue running, False if it should end its execution.
        """
        try:
            work = self.queue.get(timeout=self.max_seconds_idle)
        except queue.Empty:
            # max_seconds_idle has been exhausted, exiting
            self.end_notify()
            return False
        else:
            self._work(work)
            self.queue.task_done()
            return True