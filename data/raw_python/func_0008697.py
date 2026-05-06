def create_album(self, title=None, description=None, images=None,
                     cover=None):
        """
        Create a new Album.

        :param title: The title of the album.
        :param description: The albums description.
        :param images: A list of the images that will be added to the album
            after it's created.  Can be Image objects, ids or a combination of
            the two.  Images that you cannot add (non-existing or not owned by
            you) will not cause exceptions, but fail silently.
        :param cover: The id of the image you want as the albums cover image.

        :returns: The newly created album.
        """
        url = self._base_url + "/3/album/"
        payload = {'ids': images, 'title': title,
                   'description': description, 'cover': cover}
        resp = self._send_request(url, params=payload, method='POST')
        return Album(resp, self, has_fetched=False)