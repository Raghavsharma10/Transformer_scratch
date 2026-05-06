def get_memes_gallery(self, sort='viral', window='week', limit=None):
        """
        Return a list of gallery albums/images submitted to the memes gallery

        The url for the memes gallery is: http://imgur.com/g/memes

        :param sort: viral | time | top - defaults to viral
        :param window: Change the date range of the request if the section is
            "top", day | week | month | year | all, defaults to week.
        :param limit: The number of items to return.
        """
        url = (self._base_url + "/3/gallery/g/memes/{0}/{1}/{2}".format(
               sort, window, '{}'))
        resp = self._send_request(url, limit=limit)
        return [_get_album_or_image(thing, self) for thing in resp]