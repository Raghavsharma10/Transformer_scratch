def list(self, allJobs=False):
        """
        Return list of jobs at this endpoint.

        Call get(allJobs=True) to see all jobs, not just the ones managed by this Client
        """

        jobs = self.server.call('get', '/job')

        return [Job(job['id'], self.server) for job in jobs if allJobs or self._job_owned(job)]