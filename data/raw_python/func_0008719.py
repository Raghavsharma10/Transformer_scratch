def change_settings(self, bio=None, public_images=None,
                        messaging_enabled=None, album_privacy=None,
                        accepted_gallery_terms=None):
        """
        Update the settings for the user.

        :param bio: A basic description filled out by the user, is displayed in
            the gallery profile page.
        :param public_images: Set the default privacy setting of the users
            images. If True images are public, if False private.
        :param messaging_enabled: Set to True to enable messaging.
        :param album_privacy: The default privacy level of albums created by
            the user. Can be public, hidden or secret.
        :param accepted_gallery_terms: The user agreement to Imgur Gallery
            terms. Necessary before the user can submit to the gallery.
        """
        # NOTE: album_privacy should maybe be renamed to default_privacy
        # NOTE: public_images is a boolean, despite the documentation saying it
        # is a string.
        url = self._imgur._base_url + "/3/account/{0}/settings".format(self.name)
        resp = self._imgur._send_request(url, needs_auth=True, params=locals(),
                                         method='POST')
        return resp