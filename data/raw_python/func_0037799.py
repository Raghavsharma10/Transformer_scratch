def list_poll_choices_in_poll(self, poll_id):
        """
        List poll choices in a poll.

        Returns the list of PollChoices in this poll.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - poll_id
        """ID"""
        path["poll_id"] = poll_id

        self.logger.debug("GET /api/v1/polls/{poll_id}/poll_choices with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/polls/{poll_id}/poll_choices".format(**path), data=data, params=params, no_data=True)