def get_gallery(self, section='hot', sort='viral', window='day',
                    show_viral=True, limit=None):
        """
        Return a list of gallery albums and gallery images.

        :param section: hot | top | user - defaults to hot.
        :param sort: viral | time - defaults to viral.
        :param window: Change the date range of the request if the section is
            "top", day | week | month | year | all, defaults to day.
        :param show_viral: true | false - Show or hide viral images from the
            'user' section. Defaults to true.
        :param limit: The number of items to return.
        """
        url = (self._base_url + "/3/gallery/{}/{}/{}/{}?showViral="
               "{}".format(section, sort, window, '{}', show_viral))
        resp = self._send_request(url, limit=limit)
        return [_get_album_or_image(thing, self) for thing in resp]