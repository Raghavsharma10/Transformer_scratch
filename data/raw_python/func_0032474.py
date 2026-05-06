def update_ssh_key(self, name):
        # The `_ssh_key` is to avoid conflicts with MutableMapping.update.
        """
        Update (i.e., rename) the SSH key

        :param str name: the new name for the SSH key
        :return: an updated `SSHKey` object
        :rtype: SSHKey
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        return api._ssh_key(api.request(self.url, method='PUT',
                                       data={"name": name})["ssh_key"])