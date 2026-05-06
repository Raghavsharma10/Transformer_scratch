def get_auth_token(self, job_id, date_range=None):
        """Get an auth token to access protected job resources.

        https://wiki.saucelabs.com/display/DOCS/Building+Links+to+Test+Results
        """
        key = '{}:{}'.format(self.client.sauce_username,
                             self.client.sauce_access_key)
        if date_range:
            key = '{}:{}'.format(key, date_range)
        return hmac.new(key.encode('utf-8'), job_id.encode('utf-8'),
                        md5).hexdigest()