def progress(self, nextRound=True):
        """
        Check the status of the current job, activate the next job if it's finished, and return the active job

        If the current job has failed, a NutchCrawlException will be raised with no jobs attached.

        :param nextRound: whether to start jobs from the next round if the current job/round is completed.
        :return: the currently running Job, or None if no jobs are running.
        """

        currentJob = self.currentJob
        if currentJob is None:
            return currentJob

        jobInfo = currentJob.info()

        if jobInfo['state'] == 'RUNNING':
            return currentJob
        elif jobInfo['state'] == 'FINISHED':
            nextJob = self._nextJob(currentJob, nextRound)
            self.currentJob = nextJob
            return nextJob
        else:
            error = NutchCrawlException("Unexpected job state: {}".format(jobInfo['state']))
            error.current_job = currentJob
            raise NutchCrawlException