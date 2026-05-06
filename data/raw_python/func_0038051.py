def list_revisions_courses(self, url, course_id):
        """
        List revisions.

        List the revisions of a page. Callers must have update rights on the page in order to see page history.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - url
        """ID"""
        path["url"] = url

        self.logger.debug("GET /api/v1/courses/{course_id}/pages/{url}/revisions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/courses/{course_id}/pages/{url}/revisions".format(**path), data=data, params=params, all_pages=True)