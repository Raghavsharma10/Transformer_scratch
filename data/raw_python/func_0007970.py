def put(self, job, result):
        "Perform a job by a member in the pool and return the result."
        self.job.put(job)
        r = result.get()
        return r