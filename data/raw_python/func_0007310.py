def get_job_asset_url(self, job_id, filename):
        """Get details about the static assets collected for a specific job."""
        return 'https://saucelabs.com/rest/v1/{}/jobs/{}/assets/{}'.format(
            self.client.sauce_username, job_id, filename)