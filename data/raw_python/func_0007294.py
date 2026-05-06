def get_auth_string(self):
        """Create auth string from credentials."""
        auth_info = '{}:{}'.format(self.sauce_username, self.sauce_access_key)
        return base64.b64encode(auth_info.encode('utf-8')).decode('utf-8')