def cancel(self, job_ids):
        """Cancel jobs specified by a list of job ids.

        Parameters
        ----------
        list of str
            List of identifiers of jobs which should be canceled.

        Returns
        -------
        list of bool
            For each entry, True if the cancel operation is successful, otherwise False.
        """
        for job_id in job_ids:
            try:
                self.deployer.destroy(self.resources.get(job_id))
                return True
            except e:
                logger.error("Failed to cancel {}".format(repr(job_id)))
                logger.error(e)
                return False