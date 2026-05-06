def make_auth_headers(self, content_type):
        """Add authorization header."""
        headers = self.make_headers(content_type)
        headers['Authorization'] = 'Basic {}'.format(self.get_auth_string())
        return headers