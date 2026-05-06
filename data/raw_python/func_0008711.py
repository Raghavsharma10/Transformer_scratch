def get_subreddit_image(self, subreddit, id):
        """
        Return the Gallery_image with the id submitted to subreddit gallery

        :param subreddit: The subreddit the image has been submitted to.
        :param id: The id of the image we want.
        """
        url = self._base_url + "/3/gallery/r/{0}/{1}".format(subreddit, id)
        resp = self._send_request(url)
        return Gallery_image(resp, self)