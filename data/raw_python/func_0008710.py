def get_subreddit_gallery(self, subreddit, sort='time', window='top',
                              limit=None):
        """
        Return a list of gallery albums/images submitted to a subreddit.

        A subreddit is a subsection of the website www.reddit.com, where users
        can, among other things, post images.

        :param subreddit: A valid subreddit name.
        :param sort: time | top - defaults to top.
        :param window: Change the date range of the request if the section is
            "top", day | week | month | year | all, defaults to day.
        :param limit: The number of items to return.
        """
        url = (self._base_url + "/3/gallery/r/{0}/{1}/{2}/{3}".format(
               subreddit, sort, window, '{}'))
        resp = self._send_request(url, limit=limit)
        return [_get_album_or_image(thing, self) for thing in resp]