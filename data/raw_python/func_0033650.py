def get(self, path, data=None, return_fields=None):
        """Call the Infoblox device to get the obj for the data passed in

        :param str obj_reference: The object reference data
        :param dict data: The data for the get request
        :rtype: requests.Response

        """
        return self.session.get(self._request_url(path, return_fields),
                                data=json.dumps(data),
                                auth=self.auth, verify=False)