def upload(self, fp):
        """Upload image data to the image service.

        Call this, rather than :func:`from_file`, you don't want to
        create an attachment of the image.

        :param file fp: a file object containing binary image data
        :return: the URLs for the image uploaded
        :rtype: dict
        """
        url = utils.urljoin(self.url, 'pictures')
        response = self.session.post(url, data=fp.read())
        image_urls = response.data
        return image_urls