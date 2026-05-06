def send(self, data):
        """ Send data over scgi to URL and get response.
        """
        start = time.time()
        try:
            scgi_resp = ''.join(self.transport.send(_encode_payload(data)))
        finally:
            self.latency = time.time() - start

        resp, self.resp_headers = _parse_response(scgi_resp)
        return resp