def get_gallery_image(self, id):
        """
        Return the gallery image matching the id.

        Note that an image's id is different from it's id as a gallery image.
        This makes it possible to remove an image from the gallery and setting
        it's privacy setting as secret, without compromising it's secrecy.
        """
        url = self._base_url + "/3/gallery/image/{0}".format(id)
        resp = self._send_request(url)
        return Gallery_image(resp, self)