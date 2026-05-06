def create_single_poll_submission(self, poll_id, poll_session_id, poll_submissions_poll_choice_id):
        """
        Create a single poll submission.

        Create a new poll submission for this poll session
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - poll_id
        """ID"""
        path["poll_id"] = poll_id

        # REQUIRED - PATH - poll_session_id
        """ID"""
        path["poll_session_id"] = poll_session_id

        # REQUIRED - poll_submissions[poll_choice_id]
        """The chosen poll choice for this submission."""
        data["poll_submissions[poll_choice_id]"] = poll_submissions_poll_choice_id

        self.logger.debug("POST /api/v1/polls/{poll_id}/poll_sessions/{poll_session_id}/poll_submissions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/polls/{poll_id}/poll_sessions/{poll_session_id}/poll_submissions".format(**path), data=data, params=params, no_data=True)