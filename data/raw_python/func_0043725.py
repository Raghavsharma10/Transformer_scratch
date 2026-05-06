def status(self, job_ids):
        """Get the status of a list of jobs identified by their ids.

        Parameters
        ----------
        job_ids : list of str
            Identifiers for the jobs.

        Returns
        -------
        list of int
            Status codes for each requested job.
        """
        states = []
        statuses = self.deployer.get_vm_status([self.resources.get(job_id) for job_id in job_ids])
        for status in statuses:
            states.append(translate_table.get(status.state['Name'], "PENDING"))
        return states