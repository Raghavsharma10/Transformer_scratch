def run_later(self, callable_, timeout, *args, **kwargs):
        """Schedules the specified callable for delayed execution.

        Returns a TimerTask instance that can be used to cancel pending
        execution.
        """

        self.lock.acquire()
        try:
            if self.die:
                raise RuntimeError('This timer has been shut down and '
                                   'does not accept new jobs.')

            job = TimerTask(callable_, *args, **kwargs)
            self._jobs.append((job, time.time() + timeout))
            self._jobs.sort(key=lambda j: j[1])  # sort on time
            self.lock.notify()

            return job
        finally:
            self.lock.release()