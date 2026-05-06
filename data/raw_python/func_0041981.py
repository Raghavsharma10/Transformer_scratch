def _send_http_request(self, xml_request):
        """ Send a request via HTTP protocol.

            Args:
                xml_request -- A fully formed xml request string for the CPS.

            Returns:
                The raw xml response string.
        """
        headers = {"Host": self._host, "Content-Type": "text/xml", "Recipient": self._storage}
        try: # Retry once if failed in case the socket has just gone bad.
            self._connection.request("POST", self._selector_url, xml_request, headers)
            response = self._connection.getresponse()
        except (httplib.CannotSendRequest, httplib.BadStatusLine):
            Debug.warn("\nRestarting socket, resending message!")
            self._open_connection()
            self._connection.request("POST", self._selector_url, xml_request, headers)
            response = self._connection.getresponse()
        data = response.read()
        return data