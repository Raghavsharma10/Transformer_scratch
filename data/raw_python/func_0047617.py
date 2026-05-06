def get(self, mac):
        """Get data from API as instance of ResponseModel.

            Keyword arguments:
            mac -- MAC address or OUI for searching
        """

        data = {
            self._FORMAT_F: 'json',
            self._SEARCH_F: mac
        }

        response = self.__decode_str(self.__call_api(self.__url, data), 'utf-8')

        if len(response) > 0:
            return self.__parse(response)
        raise EmptyResponseException()