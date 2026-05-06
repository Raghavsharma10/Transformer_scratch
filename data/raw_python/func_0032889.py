def create_tag(self, name):
        """
        .. versionadded:: 0.2.0

        Add a new tag resource to the account

        :param str name: the name of the new tag
        :rtype: Tag
        :raises DOAPIError: if the API endpoint replies with an error
        """
        return self._tag(self.request('/v2/tags', method='POST', data={
            "name": name,
        })["tag"])