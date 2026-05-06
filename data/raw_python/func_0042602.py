def add(self, value, timeout=None, force=False):
        """
        :param value:  ADDED THE THE QUEUE
        :param timeout:  HOW LONG TO WAIT FOR QUEUE TO NOT BE FULL
        :param force:  ADD TO QUEUE, EVEN IF FULL (USE ONLY WHEN CONSUMER IS RETURNING WORK TO THE QUEUE)
        :return: self
        """
        with self.lock:
            if value is THREAD_STOP:
                # INSIDE THE lock SO THAT EXITING WILL RELEASE wait()
                self.queue.append(value)
                self.closed.go()
                return

            if not force:
                self._wait_for_queue_space(timeout=timeout)
            if self.closed and not self.allow_add_after_close:
                Log.error("Do not add to closed queue")
            else:
                if self.unique:
                    if value not in self.queue:
                        self.queue.append(value)
                else:
                    self.queue.append(value)
        return self