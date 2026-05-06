def get(self, uri):
        """ Send a request to given uri. """
        return self.send_request(
            "{0}://{1}:{2}{3}{4}".format(
                self.get_protocol(),
                self.host,
                self.port,
                uri,
                self.client_id
            )
        )