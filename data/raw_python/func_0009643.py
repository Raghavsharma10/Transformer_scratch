def from_file(self, fp):
        """Create a new image attachment from an image file.

        :param file fp: a file object containing binary image data
        :return: an image attachment
        :rtype: :class:`~groupy.api.attachments.Image`
        """
        image_urls = self.upload(fp)
        return Image(image_urls['url'], source_url=image_urls['picture_url'])