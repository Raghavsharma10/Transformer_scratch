def edit_assignment_group(self, course_id, assignment_group_id):
        """
        Edit an Assignment Group.

        Modify an existing Assignment Group.
        Accepts the same parameters as Assignment Group creation
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - assignment_group_id
        """ID"""
        path["assignment_group_id"] = assignment_group_id

        self.logger.debug("PUT /api/v1/courses/{course_id}/assignment_groups/{assignment_group_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/assignment_groups/{assignment_group_id}".format(**path), data=data, params=params, single_item=True)