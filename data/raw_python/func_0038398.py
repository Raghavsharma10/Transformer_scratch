def get_single_poll_submission(self, id, poll_id, poll_session_id):
        """
        Get a single poll submission.

        Returns the poll submission with the given id
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

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        self.logger.debug("GET /api/v1/polls/{poll_id}/poll_sessions/{poll_session_id}/poll_submissions/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/polls/{poll_id}/poll_sessions/{poll_session_id}/poll_submissions/{id}".format(**path), data=data, params=params, no_data=True)