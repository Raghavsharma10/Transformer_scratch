def _status(self):
        ''' Internal: Do not call. Returns the status list for a list of job_ids

        Args:
              self

        Returns:
              [status...] : Status list of all jobs
        '''

        job_id_list = ' '.join(self.resources.keys())

        jobs_missing = list(self.resources.keys())

        retcode, stdout, stderr = self.channel.execute_wait("qstat {0}".format(job_id_list), 3)
        for line in stdout.split('\n'):
            parts = line.split()
            if not parts or parts[0].upper().startswith('JOB') or parts[0].startswith('---'):
                continue
            job_id = parts[0]
            status = translate_table.get(parts[4], 'UNKNOWN')
            self.resources[job_id]['status'] = status
            jobs_missing.remove(job_id)

        # squeue does not report on jobs that are not running. So we are filling in the
        # blanks for missing jobs, we might lose some information about why the jobs failed.
        for missing_job in jobs_missing:
            if self.resources[missing_job]['status'] in ['PENDING', 'RUNNING']:
                self.resources[missing_job]['status'] = translate_table['E']