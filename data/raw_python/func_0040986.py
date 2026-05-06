def check_job(self, job_id):
        """
        Check the status of a bulk verification job.
        :param int job_id: ID of a job to check the status of.
        :return: A JobStatus object.
        """
        resp = self._call(endpoint='status', data={'job_id': job_id})
        map = {'id': 'job_id', 'status': 'status_code', 'type': 'type_code'}
        job_status_args = {map.get(k, k): v for k, v in resp.items()}
        return JobStatus(**job_status_args)