def get_outcome_results(self, course_id, include=None, outcome_ids=None, user_ids=None):
        """
        Get outcome results.

        Gets the outcome results for users and outcomes in the specified context.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - user_ids
        """If specified, only the users whose ids are given will be included in the
        results. SIS ids can be used, prefixed by "sis_user_id:".
        It is an error to specify an id for a user who is not a student in
        the context."""
        if user_ids is not None:
            params["user_ids"] = user_ids

        # OPTIONAL - outcome_ids
        """If specified, only the outcomes whose ids are given will be included in the
        results. it is an error to specify an id for an outcome which is not linked
        to the context."""
        if outcome_ids is not None:
            params["outcome_ids"] = outcome_ids

        # OPTIONAL - include
        """[String, "alignments"|"outcomes"|"outcomes.alignments"|"outcome_groups"|"outcome_links"|"outcome_paths"|"users"]
        Specify additional collections to be side loaded with the result.
        "alignments" includes only the alignments referenced by the returned
        results.
        "outcomes.alignments" includes all alignments referenced by outcomes in the
        context."""
        if include is not None:
            params["include"] = include

        self.logger.debug("GET /api/v1/courses/{course_id}/outcome_results with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/outcome_results".format(**path), data=data, params=params, no_data=True)