def perform(self):
        """
        Performs a simple HTTP request against the configured url and returns
        true if the response has a 2xx code.

        The url can be configured to use https via the "https" boolean flag
        in the config, as well as a custom HTTP method via the "method" key.

        The default is to not use https and the GET method.
        """
        if self.use_https:
            conn = client.HTTPSConnection(self.host, self.port)
        else:
            conn = client.HTTPConnection(self.host, self.port)

        conn.request(self.method, self.uri)

        response = conn.getresponse()

        conn.close()

        return bool(response.status >= 200 and response.status < 300)