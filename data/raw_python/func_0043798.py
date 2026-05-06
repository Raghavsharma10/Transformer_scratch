def cancel(self, job_ids):
        """ Cancels the jobs specified by a list of job ids

        Args:
        job_ids : [<job_id> ...]

        Returns :
        [True/False...] : If the cancel operation fails the entire list will be False.
        """
        for job in job_ids:
            logger.debug("Terminating job/proc_id : {0}".format(job))
            # Here we are assuming that for local, the job_ids are the process id's
            self._delete_deployment(job)

            self.resources[job]['status'] = 'CANCELLED'
        rets = [True for i in job_ids]

        return rets