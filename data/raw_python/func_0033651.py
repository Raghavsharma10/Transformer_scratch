def post(self, path, data):
        """Call the Infoblox device to post the obj for the data passed in

        :param str obj: The object type
        :param dict data: The data for the post
        :rtype: requests.Response

        """
        LOGGER.debug('Posting data: %r', data)
        return self.session.post(self._request_url(path),
                                 data=json.dumps(data or {}),
                                 headers=self.HEADERS, auth=self.auth,
                                 verify=False)