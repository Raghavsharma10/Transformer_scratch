def imap_unordered(self, jobs, timeout=0.5):
        """A iterator over a set of jobs.

        :param jobs: the items to pass through our function
        :param timeout: timeout between polling queues

        Results are yielded as soon as they are available in the output
        queue (up to the discretisation provided by timeout). Since the
        queues can be specified to have a maximum length, the consumption
        of both the input jobs iterable and memory use in the output
        queues are controlled.
        """
        timeout = max(timeout, 0.5)
        jobs_iter = iter(jobs)
        out_jobs = 0
        job = None
        while True:
            if not self.closed and job is None:
                # Get a job
                try:
                    job = jobs_iter.next()
                except StopIteration:
                    job = None
                    self.close()
            if job is not None:
                # Put any job
                try:
                    self.put(job, True, timeout)
                except Queue.Full:
                    pass # we'll try again next time around
                else:
                    job = None
            for result in self.get_finished():
                yield result
                
            # Input and yielded everything?
            if self.closed and self._items == 0:
                break
            sleep(timeout)