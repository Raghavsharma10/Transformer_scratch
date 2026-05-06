def list_course_nicknames(self):
        """
        List course nicknames.

        Returns all course nicknames you have set.
        """
        path = {}
        data = {}
        params = {}

        self.logger.debug("GET /api/v1/users/self/course_nicknames with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/users/self/course_nicknames".format(**path), data=data, params=params, all_pages=True)