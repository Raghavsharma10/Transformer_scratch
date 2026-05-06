def review(self, change_id, revision, review):
        """ Submit a review.

        :arg str change_id: The change ID.
        :arg str revision: The revision.
        :arg str review: The review details as a :class:`GerritReview`.

        :returns:
            JSON decoded result.

        :raises:
            requests.RequestException on timeout or connection error.

        """

        endpoint = "changes/%s/revisions/%s/review" % (change_id, revision)
        self.post(endpoint, data=str(review))