def push(self, value):
        """
        SNEAK value TO FRONT OF THE QUEUE
        """
        if self.closed and not self.allow_add_after_close:
            Log.error("Do not push to closed queue")

        with self.lock:
            self._wait_for_queue_space()
            if not self.closed:
                self.queue.appendleft(value)
        return self