def create_quiz(self, course_id, quiz_title, quiz_access_code=None, quiz_allowed_attempts=None, quiz_assignment_group_id=None, quiz_cant_go_back=None, quiz_description=None, quiz_due_at=None, quiz_hide_correct_answers_at=None, quiz_hide_results=None, quiz_ip_filter=None, quiz_lock_at=None, quiz_one_question_at_a_time=None, quiz_one_time_results=None, quiz_only_visible_to_overrides=None, quiz_published=None, quiz_quiz_type=None, quiz_scoring_policy=None, quiz_show_correct_answers=None, quiz_show_correct_answers_at=None, quiz_show_correct_answers_last_attempt=None, quiz_shuffle_answers=None, quiz_time_limit=None, quiz_unlock_at=None):
        """
        Create a quiz.

        Create a new quiz for this course.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - quiz[title]
        """The quiz title."""
        data["quiz[title]"] = quiz_title

        # OPTIONAL - quiz[description]
        """A description of the quiz."""
        if quiz_description is not None:
            data["quiz[description]"] = quiz_description

        # OPTIONAL - quiz[quiz_type]
        """The type of quiz."""
        if quiz_quiz_type is not None:
            self._validate_enum(quiz_quiz_type, ["practice_quiz", "assignment", "graded_survey", "survey"])
            data["quiz[quiz_type]"] = quiz_quiz_type

        # OPTIONAL - quiz[assignment_group_id]
        """The assignment group id to put the assignment in. Defaults to the top
        assignment group in the course. Only valid if the quiz is graded, i.e. if
        quiz_type is "assignment" or "graded_survey"."""
        if quiz_assignment_group_id is not None:
            data["quiz[assignment_group_id]"] = quiz_assignment_group_id

        # OPTIONAL - quiz[time_limit]
        """Time limit to take this quiz, in minutes. Set to null for no time limit.
        Defaults to null."""
        if quiz_time_limit is not None:
            data["quiz[time_limit]"] = quiz_time_limit

        # OPTIONAL - quiz[shuffle_answers]
        """If true, quiz answers for multiple choice questions will be randomized for
        each student. Defaults to false."""
        if quiz_shuffle_answers is not None:
            data["quiz[shuffle_answers]"] = quiz_shuffle_answers

        # OPTIONAL - quiz[hide_results]
        """Dictates whether or not quiz results are hidden from students.
        If null, students can see their results after any attempt.
        If "always", students can never see their results.
        If "until_after_last_attempt", students can only see results after their
        last attempt. (Only valid if allowed_attempts > 1). Defaults to null."""
        if quiz_hide_results is not None:
            self._validate_enum(quiz_hide_results, ["always", "until_after_last_attempt"])
            data["quiz[hide_results]"] = quiz_hide_results

        # OPTIONAL - quiz[show_correct_answers]
        """Only valid if hide_results=null
        If false, hides correct answers from students when quiz results are viewed.
        Defaults to true."""
        if quiz_show_correct_answers is not None:
            data["quiz[show_correct_answers]"] = quiz_show_correct_answers

        # OPTIONAL - quiz[show_correct_answers_last_attempt]
        """Only valid if show_correct_answers=true and allowed_attempts > 1
        If true, hides correct answers from students when quiz results are viewed
        until they submit the last attempt for the quiz.
        Defaults to false."""
        if quiz_show_correct_answers_last_attempt is not None:
            data["quiz[show_correct_answers_last_attempt]"] = quiz_show_correct_answers_last_attempt

        # OPTIONAL - quiz[show_correct_answers_at]
        """Only valid if show_correct_answers=true
        If set, the correct answers will be visible by students only after this
        date, otherwise the correct answers are visible once the student hands in
        their quiz submission."""
        if quiz_show_correct_answers_at is not None:
            data["quiz[show_correct_answers_at]"] = quiz_show_correct_answers_at

        # OPTIONAL - quiz[hide_correct_answers_at]
        """Only valid if show_correct_answers=true
        If set, the correct answers will stop being visible once this date has
        passed. Otherwise, the correct answers will be visible indefinitely."""
        if quiz_hide_correct_answers_at is not None:
            data["quiz[hide_correct_answers_at]"] = quiz_hide_correct_answers_at

        # OPTIONAL - quiz[allowed_attempts]
        """Number of times a student is allowed to take a quiz.
        Set to -1 for unlimited attempts.
        Defaults to 1."""
        if quiz_allowed_attempts is not None:
            data["quiz[allowed_attempts]"] = quiz_allowed_attempts

        # OPTIONAL - quiz[scoring_policy]
        """Required and only valid if allowed_attempts > 1.
        Scoring policy for a quiz that students can take multiple times.
        Defaults to "keep_highest"."""
        if quiz_scoring_policy is not None:
            self._validate_enum(quiz_scoring_policy, ["keep_highest", "keep_latest"])
            data["quiz[scoring_policy]"] = quiz_scoring_policy

        # OPTIONAL - quiz[one_question_at_a_time]
        """If true, shows quiz to student one question at a time.
        Defaults to false."""
        if quiz_one_question_at_a_time is not None:
            data["quiz[one_question_at_a_time]"] = quiz_one_question_at_a_time

        # OPTIONAL - quiz[cant_go_back]
        """Only valid if one_question_at_a_time=true
        If true, questions are locked after answering.
        Defaults to false."""
        if quiz_cant_go_back is not None:
            data["quiz[cant_go_back]"] = quiz_cant_go_back

        # OPTIONAL - quiz[access_code]
        """Restricts access to the quiz with a password.
        For no access code restriction, set to null.
        Defaults to null."""
        if quiz_access_code is not None:
            data["quiz[access_code]"] = quiz_access_code

        # OPTIONAL - quiz[ip_filter]
        """Restricts access to the quiz to computers in a specified IP range.
        Filters can be a comma-separated list of addresses, or an address followed by a mask
        
        Examples:
          "192.168.217.1"
          "192.168.217.1/24"
          "192.168.217.1/255.255.255.0"
        
        For no IP filter restriction, set to null.
        Defaults to null."""
        if quiz_ip_filter is not None:
            data["quiz[ip_filter]"] = quiz_ip_filter

        # OPTIONAL - quiz[due_at]
        """The day/time the quiz is due.
        Accepts times in ISO 8601 format, e.g. 2011-10-21T18:48Z."""
        if quiz_due_at is not None:
            data["quiz[due_at]"] = quiz_due_at

        # OPTIONAL - quiz[lock_at]
        """The day/time the quiz is locked for students.
        Accepts times in ISO 8601 format, e.g. 2011-10-21T18:48Z."""
        if quiz_lock_at is not None:
            data["quiz[lock_at]"] = quiz_lock_at

        # OPTIONAL - quiz[unlock_at]
        """The day/time the quiz is unlocked for students.
        Accepts times in ISO 8601 format, e.g. 2011-10-21T18:48Z."""
        if quiz_unlock_at is not None:
            data["quiz[unlock_at]"] = quiz_unlock_at

        # OPTIONAL - quiz[published]
        """Whether the quiz should have a draft state of published or unpublished.
        NOTE: If students have started taking the quiz, or there are any
        submissions for the quiz, you may not unpublish a quiz and will recieve
        an error."""
        if quiz_published is not None:
            data["quiz[published]"] = quiz_published

        # OPTIONAL - quiz[one_time_results]
        """Whether students should be prevented from viewing their quiz results past
        the first time (right after they turn the quiz in.)
        Only valid if "hide_results" is not set to "always".
        Defaults to false."""
        if quiz_one_time_results is not None:
            data["quiz[one_time_results]"] = quiz_one_time_results

        # OPTIONAL - quiz[only_visible_to_overrides]
        """Whether this quiz is only visible to overrides (Only useful if
        'differentiated assignments' account setting is on)
        Defaults to false."""
        if quiz_only_visible_to_overrides is not None:
            data["quiz[only_visible_to_overrides]"] = quiz_only_visible_to_overrides

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes".format(**path), data=data, params=params, single_item=True)