def stop_job(self, job_id):
        """Terminates a running job."""
        method = 'PUT'
        endpoint = '/rest/v1/{}/jobs/{}/stop'.format(
            self.client.sauce_username, job_id)
        return self.client.request(method, endpoint)