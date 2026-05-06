def contract(self, jobs, result):
        """
        Perform a contract on a number of jobs and block until a result is
        retrieved for each job.
        """
        for j in jobs:
            WorkerPool.put(self, j)

        r = []
        for i in xrange(len(jobs)):
            r.append(result.get())

        return r