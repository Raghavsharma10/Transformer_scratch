def make_response(self, status, content_type, response):
        """Shortcut for making a response to the client's request."""
        headers = [('Access-Control-Allow-Origin', '*'),
                   ('Access-Control-Allow-Methods', 'GET, POST, OPTIONS'),
                   ('Access-Control-Allow-Headers', 'Content-Type'),
                   ('Access-Control-Max-Age', '86400'),
                   ('Content-type', content_type)
                  ]
        self.start_response(status, headers)
        return [response.encode()]