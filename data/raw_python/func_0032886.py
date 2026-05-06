def fetch_all_images(self, type=None, private=None):
        # pylint: disable=redefined-builtin
        r"""
        Returns a generator that yields all of the images available to the
        account

        :param type: the type of images to fetch: ``"distribution"``,
            ``"application"``, or all (`None`); default: `None`
        :type type: string or None
        :param bool private: whether to only return the user's private images;
            default: return all images
        :rtype: generator of `Image`\ s
        :raises DOAPIError: if the API endpoint replies with an error
        """
        params = {}
        if type is not None:
            params["type"] = type
        if private is not None:
            params["private"] = 'true' if private else 'false'
        return map(self._image, self.paginate('/v2/images', 'images',
                                              params=params))