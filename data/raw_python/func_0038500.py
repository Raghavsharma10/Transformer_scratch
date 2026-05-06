def list_discussion_topics_courses(self, course_id, exclude_context_module_locked_topics=None, include=None, only_announcements=None, order_by=None, scope=None, search_term=None):
        """
        List discussion topics.

        Returns the paginated list of discussion topics for this course or group.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # OPTIONAL - include
        """If "all_dates" is passed, all dates associated with graded discussions'
        assignments will be included."""
        if include is not None:
            self._validate_enum(include, ["all_dates"])
            params["include"] = include

        # OPTIONAL - order_by
        """Determines the order of the discussion topic list. Defaults to "position"."""
        if order_by is not None:
            self._validate_enum(order_by, ["position", "recent_activity"])
            params["order_by"] = order_by

        # OPTIONAL - scope
        """Only return discussion topics in the given state(s). Defaults to including
        all topics. Filtering is done after pagination, so pages
        may be smaller than requested if topics are filtered.
        Can pass multiple states as comma separated string."""
        if scope is not None:
            self._validate_enum(scope, ["locked", "unlocked", "pinned", "unpinned"])
            params["scope"] = scope

        # OPTIONAL - only_announcements
        """Return announcements instead of discussion topics. Defaults to false"""
        if only_announcements is not None:
            params["only_announcements"] = only_announcements

        # OPTIONAL - search_term
        """The partial title of the discussion topics to match and return."""
        if search_term is not None:
            params["search_term"] = search_term

        # OPTIONAL - exclude_context_module_locked_topics
        """For students, exclude topics that are locked by module progression.
        Defaults to false."""
        if exclude_context_module_locked_topics is not None:
            params["exclude_context_module_locked_topics"] = exclude_context_module_locked_topics

        self.logger.debug("GET /api/v1/courses/{course_id}/discussion_topics with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/discussion_topics".format(**path), data=data, params=params, all_pages=True)