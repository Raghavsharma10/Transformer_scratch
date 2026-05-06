def create_question_group(self, quiz_id, course_id, quiz_groups_assessment_question_bank_id=None, quiz_groups_name=None, quiz_groups_pick_count=None, quiz_groups_question_points=None):
        """
        Create a question group.

        Create a new question group for this quiz
        
        <b>201 Created</b> response code is returned if the creation was successful.
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

        # OPTIONAL - quiz_groups[assessment_question_bank_id]
        """The id of the assessment question bank to pull questions from."""
        if quiz_groups_assessment_question_bank_id is not None:
            data["quiz_groups[assessment_question_bank_id]"] = quiz_groups_assessment_question_bank_id

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes/{quiz_id}/groups with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/groups".format(**path), data=data, params=params, no_data=True)