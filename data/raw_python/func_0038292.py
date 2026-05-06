def cross_list_section(self, id, new_course_id):
        """
        Cross-list a Section.

        Move the Section to another course.  The new course may be in a different account (department),
        but must belong to the same root account (institution).
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - PATH - new_course_id
        """ID"""
        path["new_course_id"] = new_course_id

        self.logger.debug("POST /api/v1/sections/{id}/crosslist/{new_course_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/sections/{id}/crosslist/{new_course_id}".format(**path), data=data, params=params, single_item=True)