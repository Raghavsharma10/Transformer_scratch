def get_raw_data(self, mac, response_format='json'):
        """Get data from API as string.

            Keyword arguments:
            mac -- MAC address or OUI for searching
            response_format -- supported types you can see on the https://macaddress.io
        """

        data = {
            self._FORMAT_F: response_format,
            self._SEARCH_F: mac
        }

        response = self.__decode_str(self.__call_api(self.__url, data), 'utf-8')

        if len(response) > 0:
            return response
        raise EmptyResponseException()