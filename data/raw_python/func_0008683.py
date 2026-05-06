def remove_images(self, images):
        """
        Remove images from the album.

        :param images: A list of the images we want to remove from the album.
            Can be Image objects, ids or a combination of the two. Images that
            you cannot remove (non-existing, not owned by you or not part of
            album) will not cause exceptions, but fail silently.
        """
        url = (self._imgur._base_url + "/3/album/{0}/"
               "remove_images".format(self._delete_or_id_hash))
        # NOTE: Returns True and everything seem to be as it should in testing.
        # Seems most likely to be upstream bug.
        params = {'ids': images}
        return self._imgur._send_request(url, params=params, method="DELETE")