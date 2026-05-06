def create_single_poll_session(self, poll_id, poll_sessions_course_id, poll_sessions_course_section_id=None, poll_sessions_has_public_results=None):
        """
        Create a single poll session.

        Create a new poll session for this poll
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - poll_id
        """ID"""
        path["poll_id"] = poll_id

        # REQUIRED - poll_sessions[course_id]
        """The id of the course this session is associated with."""
        data["poll_sessions[course_id]"] = poll_sessions_course_id

        # OPTIONAL - poll_sessions[course_section_id]
        """The id of the course section this session is associated with."""
        if poll_sessions_course_section_id is not None:
            data["poll_sessions[course_section_id]"] = poll_sessions_course_section_id

        # OPTIONAL - poll_sessions[has_public_results]
        """Whether or not results are viewable by students."""
        if poll_sessions_has_public_results is not None:
            data["poll_sessions[has_public_results]"] = poll_sessions_has_public_results

        self.logger.debug("POST /api/v1/polls/{poll_id}/poll_sessions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/polls/{poll_id}/poll_sessions".format(**path), data=data, params=params, no_data=True)