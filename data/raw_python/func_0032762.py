def update_tag(self, name):
        # The `_tag` is to avoid conflicts with MutableMapping.update.
        """
        Update (i.e., rename) the tag

        :param str name: the new name for the tag
        :return: an updated `Tag` object
        :rtype: Tag
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        return api._tag(api.request(self.url, method='PUT',
                                    data={"name": name})["tag"])