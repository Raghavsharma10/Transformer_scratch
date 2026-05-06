def update_image(self, name):
        # The `_image` is to avoid conflicts with MutableMapping.update.
        """
        Update (i.e., rename) the image

        :param str name: the new name for the image
        :return: an updated `Image` object
        :rtype: Image
        :raises DOAPIError: if the API endpoint replies with an error
        """
        api = self.doapi_manager
        return api._image(api.request(self.url, method='PUT',
                                               data={"name": name})["image"])