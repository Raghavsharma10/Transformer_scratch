def submit_to_gallery(self, title, bypass_terms=False):
        """
        Add this to the gallery.

        Require that the authenticated user has accepted gallery terms and
        verified their email.

        :param title: The title of the new gallery item.
        :param bypass_terms: If the user has not accepted Imgur's terms yet,
            this method will return an error. Set this to True to by-pass the
            terms.
        """
        url = self._imgur._base_url + "/3/gallery/{0}".format(self.id)
        payload = {'title': title, 'terms': '1' if bypass_terms else '0'}
        self._imgur._send_request(url, needs_auth=True, params=payload,
                                  method='POST')
        item = self._imgur.get_gallery_album(self.id)
        _change_object(self, item)
        return self