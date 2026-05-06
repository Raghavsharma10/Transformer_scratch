def get_data(self, params={}):
        """
        Make the request and return the deserialided JSON from the response

        :param params: Dictionary mapping (string) query parameters to values
        :type params: dict
        :return: JSON object with the data fetched from that URL as a
                 JSON-format object.
        :rtype: (dict or array)

        """
        if self and hasattr(self, 'proxies') and self.proxies is not None:
            response = requests.request('GET',
                                        url=PVWatts.PVWATTS_QUERY_URL,
                                        params=params,
                                        headers={'User-Agent': ''.join(
                                                 ['pypvwatts/', VERSION,
                                                  ' (Python)'])},
                                        proxies=self.proxies)
        else:
            response = requests.request('GET',
                                        url=PVWatts.PVWATTS_QUERY_URL,
                                        params=params,
                                        headers={'User-Agent': ''.join(
                                                 ['pypvwatts/', VERSION,
                                                  ' (Python)'])})

        if response.status_code == 403:
            raise PVWattsError("Forbidden, 403")
        return response.json()