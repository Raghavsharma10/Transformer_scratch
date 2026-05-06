def _send_request(self, xml_request):
        """ Send the prepared XML request block to the CPS using the corect protocol.

            Args:
                xml_request -- A fully formed xml request string for the CPS.

            Returns:
                The raw xml response string.

            Raises:
                ConnectionError -- Can't establish a connection with the server.
        """
        if self._scheme == 'http':
            return self._send_http_request(xml_request)
        else:
            return self._send_socket_request(xml_request)