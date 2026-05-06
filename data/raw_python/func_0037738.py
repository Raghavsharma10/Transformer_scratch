def set_course_nickname(self, nickname, course_id):
        """
        Set course nickname.

        Set a nickname for the given course. This will replace the course's name
        in output of API calls you make subsequently, as well as in selected
        places in the Canvas web user interface.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - nickname
        """The nickname to set.  It must be non-empty and shorter than 60 characters."""
        data["nickname"] = nickname

        self.logger.debug("PUT /api/v1/users/self/course_nicknames/{course_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/users/self/course_nicknames/{course_id}".format(**path), data=data, params=params, single_item=True)