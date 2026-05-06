def create_ssh_key(self, name, public_key, **kwargs):
        """
        Add a new SSH public key resource to the account

        :param str name: the name to give the new SSH key resource
        :param str public_key: the text of the public key to register, in the
            form used by :file:`authorized_keys` files
        :param kwargs: additional fields to include in the API request
        :return: the new SSH key resource
        :rtype: SSHKey
        :raises DOAPIError: if the API endpoint replies with an error
        """
        data = {"name": name, "public_key": public_key}
        data.update(kwargs)
        return self._ssh_key(self.request('/v2/account/keys', method='POST',
                                          data=data)["ssh_key"])