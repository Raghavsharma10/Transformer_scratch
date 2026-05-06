def create_single_poll_choice(self, poll_id, poll_choices_text, poll_choices_is_correct=None, poll_choices_position=None):
        """
        Create a single poll choice.

        Create a new poll choice for this poll
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - poll_id
        """ID"""
        path["poll_id"] = poll_id

        # REQUIRED - poll_choices[text]
        """The descriptive text of the poll choice."""
        data["poll_choices[text]"] = poll_choices_text

        # OPTIONAL - poll_choices[is_correct]
        """Whether this poll choice is considered correct or not. Defaults to false."""
        if poll_choices_is_correct is not None:
            data["poll_choices[is_correct]"] = poll_choices_is_correct

        # OPTIONAL - poll_choices[position]
        """The order this poll choice should be returned in the context it's sibling poll choices."""
        if poll_choices_position is not None:
            data["poll_choices[position]"] = poll_choices_position

        self.logger.debug("POST /api/v1/polls/{poll_id}/poll_choices with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/polls/{poll_id}/poll_choices".format(**path), data=data, params=params, no_data=True)