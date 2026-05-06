def download(self, image, url_field='url', suffix=None):
        """Download the binary data of an image attachment.

        :param image: an image attachment
        :type image: :class:`~groupy.api.attachments.Image`
        :param str url_field: the field of the image with the right URL
        :param str suffix: an optional URL suffix
        :return: binary image data
        :rtype: bytes
        """
        url = getattr(image, url_field)
        if suffix is not None:
            url = '.'.join(url, suffix)
        response = self.session.get(url)
        return response.content