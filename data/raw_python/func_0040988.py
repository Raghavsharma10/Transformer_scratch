def retrieve_job(self, job_id):
        """
        Result of a completed bulk verification job.
        :param int job_id: ID of a job to retrieve the results for.
        :return: A list of VerifiedEmail objects.
        """
        warnings.warn('Use results generator method instead of retrieve_job which returns a list', UserWarning)
        return list(self.results(job_id))