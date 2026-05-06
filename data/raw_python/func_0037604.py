def list_group_categories_for_context_courses(self, course_id):
        """
        List group categories for a context.

        Returns a list of group categories in a context
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        self.logger.debug("GET /api/v1/courses/{course_id}/group_categories with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/group_categories".format(**path), data=data, params=params, all_pages=True)