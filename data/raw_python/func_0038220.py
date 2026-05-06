def create_single_poll(self, polls_question, polls_description=None):
        """
        Create a single poll.

        Create a new poll for the current user
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - polls[question]
        """The title of the poll."""
        data["polls[question]"] = polls_question

        # OPTIONAL - polls[description]
        """A brief description or instructions for the poll."""
        if polls_description is not None:
            data["polls[description]"] = polls_description

        self.logger.debug("POST /api/v1/polls with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/polls".format(**path), data=data, params=params, no_data=True)