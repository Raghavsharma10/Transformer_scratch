def downvote(self):
        """
        Dislike this.

        A downvote will replace a neutral vote or an upvote. Downvoting
        something the authenticated user has already downvoted will set the
        vote to neutral.
        """
        url = self._imgur._base_url + "/3/gallery/{0}/vote/down".format(self.id)
        return self._imgur._send_request(url, needs_auth=True, method='POST')