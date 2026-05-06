def data(self):
        """Returns the data sent with the request."""

        # the environment variable CONTENT_LENGTH may be empty or missing
        try:
            request_body_size = int(self.environ.get('CONTENT_LENGTH', 0))
        except (ValueError):
            request_body_size = 0

        data = self.environ['wsgi.input'].read(request_body_size)

        return data