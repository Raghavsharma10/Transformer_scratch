def list_uncollated_submission_versions(self, course_id, ascending=None, assignment_id=None, user_id=None):
        """
        List uncollated submission versions.

        Gives a paginated, uncollated list of submission versions for all matching
        submissions in the context. This SubmissionVersion objects will not include
        the +new_grade+ or +previous_grade+ keys, only the +grade+; same for
        +graded_at+ and +grader+.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """The id of the contextual course for this API call"""
        path["course_id"] = course_id

        # OPTIONAL - assignment_id
        """The ID of the assignment for which you want to see submissions. If
        absent, versions of submissions from any assignment in the course are
        included."""
        if assignment_id is not None:
            params["assignment_id"] = assignment_id

        # OPTIONAL - user_id
        """The ID of the user for which you want to see submissions. If absent,
        versions of submissions from any user in the course are included."""
        if user_id is not None:
            params["user_id"] = user_id

        # OPTIONAL - ascending
        """Returns submission versions in ascending date order (oldest first). If
        absent, returns submission versions in descending date order (newest
        first)."""
        if ascending is not None:
            params["ascending"] = ascending

        self.logger.debug("GET /api/v1/courses/{course_id}/gradebook_history/feed with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/gradebook_history/feed".format(**path), data=data, params=params, all_pages=True)