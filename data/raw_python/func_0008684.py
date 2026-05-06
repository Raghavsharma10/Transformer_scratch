def set_images(self, images):
        """
        Set the images in this album.

        :param images: A list of the images we want the album to contain.
            Can be Image objects, ids or a combination of the two. Images that
            images that you cannot set (non-existing or not owned by you) will
            not cause exceptions, but fail silently.
        """
        url = (self._imgur._base_url + "/3/album/"
               "{0}/".format(self._delete_or_id_hash))
        params = {'ids': images}
        return self._imgur._send_request(url, needs_auth=True, params=params,
                                         method="POST")