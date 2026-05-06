def get_outcome_result_rollups(self, course_id, aggregate=None, include=None, outcome_ids=None, user_ids=None):
        """
        Get outcome result rollups.

        Gets the outcome rollups for the users and outcomes in the specified
        context.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - aggregate
        """If specified, instead of returning one rollup for each user, all the user
        rollups will be combined into one rollup for the course that will contain
        the average rollup score for each outcome."""
        if aggregate is not None:
            self._validate_enum(aggregate, ["course"])
            params["aggregate"] = aggregate

        # OPTIONAL - user_ids
        """If specified, only the users whose ids are given will be included in the
        results or used in an aggregate result. it is an error to specify an id
        for a user who is not a student in the context"""
        if user_ids is not None:
            params["user_ids"] = user_ids

        # OPTIONAL - outcome_ids
        """If specified, only the outcomes whose ids are given will be included in the
        results. it is an error to specify an id for an outcome which is not linked
        to the context."""
        if outcome_ids is not None:
            params["outcome_ids"] = outcome_ids

        # OPTIONAL - include
        """[String, "courses"|"outcomes"|"outcomes.alignments"|"outcome_groups"|"outcome_links"|"outcome_paths"|"users"]
        Specify additional collections to be side loaded with the result."""
        if include is not None:
            params["include"] = include

        self.logger.debug("GET /api/v1/courses/{course_id}/outcome_rollups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/outcome_rollups".format(**path), data=data, params=params, no_data=True)