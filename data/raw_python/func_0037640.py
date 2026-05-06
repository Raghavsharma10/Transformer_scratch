def list_all_courses(self, open_enrollment_only=None, public_only=None, search=None):
        """
        List all courses.

        List all courses visible in the public index
        """
        path = {}
        data = {}
        params = {}

        # OPTIONAL - search
        """Search terms used for matching users/courses/groups (e.g. "bob smith"). If
        multiple terms are given (separated via whitespace), only results matching
        all terms will be returned."""
        if search is not None:
            params["search"] = search

        # OPTIONAL - public_only
        """Only return courses with public content. Defaults to false."""
        if public_only is not None:
            params["public_only"] = public_only

        # OPTIONAL - open_enrollment_only
        """Only return courses that allow self enrollment. Defaults to false."""
        if open_enrollment_only is not None:
            params["open_enrollment_only"] = open_enrollment_only

        self.logger.debug("GET /api/v1/search/all_courses with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/search/all_courses".format(**path), data=data, params=params, no_data=True)