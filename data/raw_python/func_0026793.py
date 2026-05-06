def process_callback(self, block=True):
        """Dispatch a single callback in the current thread.

        :param boolean block: If True, blocks waiting for a callback to come.
        :return: True if a callback was processed; otherwise False.
        """
        try:
            (callback, args) = self._queue.get(block=block)
            try:
                callback(*args)
            finally:
                self._queue.task_done()
        except queue.Empty:
            return False
        return True