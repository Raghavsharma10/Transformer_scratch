def nextRound(self):
        """
        Execute all jobs in the current round and return when they have finished.

        If a job fails, a NutchCrawlException will be raised, with all completed jobs from this round attached
        to the exception.

        :return: a list of all completed Jobs
        """

        finishedJobs = []
        if self.currentJob is None:
            self.currentJob = self.jobClient.create('GENERATE')

        activeJob = self.progress(nextRound=False)
        while activeJob:
            oldJob = activeJob
            activeJob = self.progress(nextRound=False)  # updates self.currentJob
            if oldJob and oldJob != activeJob:
                finishedJobs.append(oldJob)
            sleep(self.sleepTime)
        self.currentRound += 1
        return finishedJobs