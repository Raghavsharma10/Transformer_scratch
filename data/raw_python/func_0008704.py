def get_gallery_album(self, id):
        """
        Return the gallery album matching the id.

        Note that an album's id is different from it's id as a gallery album.
        This makes it possible to remove an album from the gallery and setting
        it's privacy setting as secret, without compromising it's secrecy.
        """
        url = self._base_url + "/3/gallery/album/{0}".format(id)
        resp = self._send_request(url)
        return Gallery_album(resp, self)