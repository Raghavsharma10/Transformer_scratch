def get_vendor(self, mac):
        """Get vendor company name.

            Keyword arguments:
            mac -- MAC address or OUI for searching
        """

        data = {
            self._SEARCH_F: mac,
            self._FORMAT_F: self._VERBOSE_T
        }

        response = self.__decode_str(self.__call_api(self.__url, data), 'utf-8')

        return response