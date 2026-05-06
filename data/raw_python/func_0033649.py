def delete(self, path):
        """Call the Infoblox device to delete the ref

        :param str ref: The reference id
        :rtype: requests.Response

        """
        return self.session.delete(self._request_url(path),
                                   auth=self.auth, verify=False)