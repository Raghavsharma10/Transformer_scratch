def update(self, title=None, description=None, images=None, cover=None,
               layout=None, privacy=None):
        """
        Update the album's information.

        Arguments with the value None will retain their old values.

        :param title: The title of the album.
        :param description: A description of the album.
        :param images: A list of the images we want the album to contain.
            Can be Image objects, ids or a combination of the two. Images that
            images that you cannot set (non-existing or not owned by you) will
            not cause exceptions, but fail silently.
        :param privacy: The albums privacy level, can be public, hidden or
            secret.
        :param cover: The id of the cover image.
        :param layout: The way the album is displayed, can be blog, grid,
            horizontal or vertical.
        """
        url = (self._imgur._base_url + "/3/album/"
               "{0}".format(self._delete_or_id_hash))
        is_updated = self._imgur._send_request(url, params=locals(),
                                               method='POST')
        if is_updated:
            self.title = title or self.title
            self.description = description or self.description
            self.layout = layout or self.layout
            self.privacy = privacy or self.privacy
            if cover is not None:
                self.cover = (cover if isinstance(cover, Image)
                              else Image({'id': cover}, self._imgur,
                                         has_fetched=False))
            if images:
                self.images = [img if isinstance(img, Image) else
                               Image({'id': img}, self._imgur, False)
                               for img in images]
        return is_updated