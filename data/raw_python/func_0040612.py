def generate_headers(self, client_type, client_id, secret):
        """
        generate_headers is used to generate the headers automatically for your http request

        :param client_type (str): remoteci or feeder
        :param client_id (str): remoteci or feeder id
        :param secret (str): api secret
        :return: Authorization headers (dict)
        """

        self.request.add_header(self.dci_datetime_header, self.dci_datetime_str)
        signature = self._sign(secret)
        return self.request.build_headers(client_type, client_id, signature)