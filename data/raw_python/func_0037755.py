def update_question_group(self, id, quiz_id, course_id, quiz_groups_name=None, quiz_groups_pick_count=None, quiz_groups_question_points=None):
        """
        Update a question group.

        Update a question group
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - quiz_id
        """ID"""
        path["quiz_id"] = quiz_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - quiz_groups[name]
        """The name of the question group."""
        if quiz_groups_name is not None:
            data["quiz_groups[name]"] = quiz_groups_name

        # OPTIONAL - quiz_groups[pick_count]
        """The number of questions to randomly select for this group."""
        if quiz_groups_pick_count is not None:
            data["quiz_groups[pick_count]"] = quiz_groups_pick_count

        # OPTIONAL - quiz_groups[question_points]
        """The number of points to assign to each question in the group."""
        if quiz_groups_question_points is not None:
            data["quiz_groups[question_points]"] = quiz_groups_question_points

        self.logger.debug("PUT /api/v1/courses/{course_id}/quizzes/{quiz_id}/groups/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/groups/{id}".format(**path), data=data, params=params, no_data=True)