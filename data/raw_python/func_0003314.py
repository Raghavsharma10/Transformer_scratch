async def redirect(self, path, status = 302):
        """
        Redirect this request with 3xx status
        """
        location = urljoin(urlunsplit((b'https' if self.https else b'http',
                                                                     self.host,
                                                                     quote_from_bytes(self.path).encode('ascii'),
                                                                     '',
                                                                     ''
                                                                     )), path)
        self.start_response(status, [(b'Location', location)])
        await self.write(b'<a href="' + self.escape(location, True) + b'">' + self.escape(location) + b'</a>')
        await self.flush(True)