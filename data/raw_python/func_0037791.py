def grade_or_comment_on_submission_courses(self, user_id, course_id, assignment_id, comment_file_ids=None, comment_group_comment=None, comment_media_comment_id=None, comment_media_comment_type=None, comment_text_comment=None, include_visibility=None, rubric_assessment=None, submission_excuse=None, submission_posted_grade=None):
        """
        Grade or comment on a submission.

        Comment on and/or update the grading for a student's assignment submission.
        If any submission or rubric_assessment arguments are provided, the user
        must have permission to manage grades in the appropriate context (course or
        section).
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - assignment_id
        """ID"""
        path["assignment_id"] = assignment_id

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # OPTIONAL - comment[text_comment]
        """Add a textual comment to the submission."""
        if comment_text_comment is not None:
            data["comment[text_comment]"] = comment_text_comment

        # OPTIONAL - comment[group_comment]
        """Whether or not this comment should be sent to the entire group (defaults
        to false). Ignored if this is not a group assignment or if no text_comment
        is provided."""
        if comment_group_comment is not None:
            data["comment[group_comment]"] = comment_group_comment

        # OPTIONAL - comment[media_comment_id]
        """Add an audio/video comment to the submission. Media comments can be added
        via this API, however, note that there is not yet an API to generate or
        list existing media comments, so this functionality is currently of
        limited use."""
        if comment_media_comment_id is not None:
            data["comment[media_comment_id]"] = comment_media_comment_id

        # OPTIONAL - comment[media_comment_type]
        """The type of media comment being added."""
        if comment_media_comment_type is not None:
            self._validate_enum(comment_media_comment_type, ["audio", "video"])
            data["comment[media_comment_type]"] = comment_media_comment_type

        # OPTIONAL - comment[file_ids]
        """Attach files to this comment that were previously uploaded using the
        Submission Comment API's files action"""
        if comment_file_ids is not None:
            data["comment[file_ids]"] = comment_file_ids

        # OPTIONAL - include[visibility]
        """Whether this assignment is visible to the owner of the submission"""
        if include_visibility is not None:
            data["include[visibility]"] = include_visibility

        # OPTIONAL - submission[posted_grade]
        """Assign a score to the submission, updating both the "score" and "grade"
        fields on the submission record. This parameter can be passed in a few
        different formats:
        
        points:: A floating point or integral value, such as "13.5". The grade
          will be interpreted directly as the score of the assignment.
          Values above assignment.points_possible are allowed, for awarding
          extra credit.
        percentage:: A floating point value appended with a percent sign, such as
           "40%". The grade will be interpreted as a percentage score on the
           assignment, where 100% == assignment.points_possible. Values above 100%
           are allowed, for awarding extra credit.
        letter grade:: A letter grade, following the assignment's defined letter
           grading scheme. For example, "A-". The resulting score will be the high
           end of the defined range for the letter grade. For instance, if "B" is
           defined as 86% to 84%, a letter grade of "B" will be worth 86%. The
           letter grade will be rejected if the assignment does not have a defined
           letter grading scheme. For more fine-grained control of scores, pass in
           points or percentage rather than the letter grade.
        "pass/complete/fail/incomplete":: A string value of "pass" or "complete"
           will give a score of 100%. "fail" or "incomplete" will give a score of
           0.
        
        Note that assignments with grading_type of "pass_fail" can only be
        assigned a score of 0 or assignment.points_possible, nothing inbetween. If
        a posted_grade in the "points" or "percentage" format is sent, the grade
        will only be accepted if the grade equals one of those two values."""
        if submission_posted_grade is not None:
            data["submission[posted_grade]"] = submission_posted_grade

        # OPTIONAL - submission[excuse]
        """Sets the "excused" status of an assignment."""
        if submission_excuse is not None:
            data["submission[excuse]"] = submission_excuse

        # OPTIONAL - rubric_assessment
        """Assign a rubric assessment to this assignment submission. The
        sub-parameters here depend on the rubric for the assignment. The general
        format is, for each row in the rubric:
        
        The points awarded for this row.
          rubric_assessment[criterion_id][points]
        
        Comments to add for this row.
          rubric_assessment[criterion_id][comments]
        
        For example, if the assignment rubric is (in JSON format):
          !!!javascript
          [
            {
              'id': 'crit1',
              'points': 10,
              'description': 'Criterion 1',
              'ratings':
              [
                { 'description': 'Good', 'points': 10 },
                { 'description': 'Poor', 'points': 3 }
              ]
            },
            {
              'id': 'crit2',
              'points': 5,
              'description': 'Criterion 2',
              'ratings':
              [
                { 'description': 'Complete', 'points': 5 },
                { 'description': 'Incomplete', 'points': 0 }
              ]
            }
          ]
        
        Then a possible set of values for rubric_assessment would be:
            rubric_assessment[crit1][points]=3&rubric_assessment[crit2][points]=5&rubric_assessment[crit2][comments]=Well%20Done."""
        if rubric_assessment is not None:
            data["rubric_assessment"] = rubric_assessment

        self.logger.debug("PUT /api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{user_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/courses/{course_id}/assignments/{assignment_id}/submissions/{user_id}".format(**path), data=data, params=params, no_data=True)