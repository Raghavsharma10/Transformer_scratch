def status(self, job_ids):
        '''  Get the status of a list of jobs identified by their ids.

        Args:
            - job_ids (List of ids) : List of identifiers for the jobs

        Returns:
            - List of status codes.

        '''

        logging.debug("Checking status of : {0}".format(job_ids))
        for job_id in self.resources:
            poll_code = self.resources[job_id]['proc'].poll()
            if self.resources[job_id]['status'] in ['COMPLETED', 'FAILED']:
                continue

            if poll_code is None:
                self.resources[job_id]['status'] = 'RUNNING'
            elif poll_code == 0 and self.resources[job_id]['status'] != 'RUNNING':
                self.resources[job_id]['status'] = 'COMPLETED'
            elif poll_code < 0 and self.resources[job_id]['status'] != 'RUNNING':
                self.resources[job_id]['status'] = 'FAILED'

        return [self.resources[jid]['status'] for jid in job_ids]