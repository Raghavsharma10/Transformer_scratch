def add_images(self, images):
        """
        Add images to the album.

        :param images: A list of the images we want to add to the album. Can be
            Image objects, ids or a combination of the two.  Images that you
            cannot add (non-existing or not owned by you) will not cause
            exceptions, but fail silently.
        """
        url = self._imgur._base_url + "/3/album/{0}/add".format(self.id)
        params = {'ids': images}
        return self._imgur._send_request(url, needs_auth=True, params=params,
                                         method="POST")