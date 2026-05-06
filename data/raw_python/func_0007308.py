def update_job(self, job_id, build=None, custom_data=None,
                   name=None, passed=None, public=None, tags=None):
        """Edit an existing job."""
        method = 'PUT'
        endpoint = '/rest/v1/{}/jobs/{}'.format(self.client.sauce_username,
                                                job_id)
        data = {}
        if build is not None:
            data['build'] = build
        if custom_data is not None:
            data['custom-data'] = custom_data
        if name is not None:
            data['name'] = name
        if passed is not None:
            data['passed'] = passed
        if public is not None:
            data['public'] = public
        if tags is not None:
            data['tags'] = tags
        body = json.dumps(data)
        return self.client.request(method, endpoint, body=body)