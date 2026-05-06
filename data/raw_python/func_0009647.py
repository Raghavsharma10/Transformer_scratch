def download_large(self, image, url_field='url'):
        """Downlaod the binary data of an image attachment at large size.

        :param str url_field: the field of the image with the right URL
        :return: binary image data
        :rtype: bytes

        """
        return self.download(image, url_field=url_field, suffix='large')