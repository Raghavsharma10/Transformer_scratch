def pop(self, timeout=None):
        """
        :param timeout: OPTIONAL DURATION
        :return: None, IF timeout PASSES
        """
        with self.lock:
            while not self.please_stop:
                if self.db.status.end > self.start:
                    value = self.db[str(self.start)]
                    self.start += 1
                    return value

                if timeout is not None:
                    with suppress_exception:
                        self.lock.wait(timeout=timeout)
                        if self.db.status.end <= self.start:
                            return None
                else:
                    self.lock.wait()

            DEBUG and Log.note("persistent queue already stopped")
            return THREAD_STOP