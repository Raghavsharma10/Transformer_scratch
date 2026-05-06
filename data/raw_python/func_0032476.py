def should_close(self):
        """
        Check whether the HTTP connection of this request should be closed
        after the request is finished.

        We check for the `Connection` HTTP header and for the HTTP Version
        (only `HTTP/1.1` supports keep-alive.
        """
        if self.headers.get('connection') == 'close':
            return True
        elif 'content-length' in self.headers or \
            self.headers.get('METHOD') in ['HEAD', 'GET']:
            return self.headers.get('connection') != 'keep-alive'
        elif self.headers.get('VERSION') == 'HTTP/1.0':
            return True
        else:
            return False