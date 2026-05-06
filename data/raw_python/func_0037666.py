def create_single_quiz_question(self, quiz_id, course_id, question_answers=None, question_correct_comments=None, question_incorrect_comments=None, question_neutral_comments=None, question_points_possible=None, question_position=None, question_question_name=None, question_question_text=None, question_question_type=None, question_quiz_group_id=None, question_text_after_answers=None):
        """
        Create a single quiz question.

        Create a new quiz question for this quiz
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

        # OPTIONAL - question[question_name]
        """The name of the question."""
        if question_question_name is not None:
            data["question[question_name]"] = question_question_name

        # OPTIONAL - question[question_text]
        """The text of the question."""
        if question_question_text is not None:
            data["question[question_text]"] = question_question_text

        # OPTIONAL - question[quiz_group_id]
        """The id of the quiz group to assign the question to."""
        if question_quiz_group_id is not None:
            data["question[quiz_group_id]"] = question_quiz_group_id

        # OPTIONAL - question[question_type]
        """The type of question. Multiple optional fields depend upon the type of question to be used."""
        if question_question_type is not None:
            self._validate_enum(question_question_type, ["calculated_question", "essay_question", "file_upload_question", "fill_in_multiple_blanks_question", "matching_question", "multiple_answers_question", "multiple_choice_question", "multiple_dropdowns_question", "numerical_question", "short_answer_question", "text_only_question", "true_false_question"])
            data["question[question_type]"] = question_question_type

        # OPTIONAL - question[position]
        """The order in which the question will be displayed in the quiz in relation to other questions."""
        if question_position is not None:
            data["question[position]"] = question_position

        # OPTIONAL - question[points_possible]
        """The maximum amount of points received for answering this question correctly."""
        if question_points_possible is not None:
            data["question[points_possible]"] = question_points_possible

        # OPTIONAL - question[correct_comments]
        """The comment to display if the student answers the question correctly."""
        if question_correct_comments is not None:
            data["question[correct_comments]"] = question_correct_comments

        # OPTIONAL - question[incorrect_comments]
        """The comment to display if the student answers incorrectly."""
        if question_incorrect_comments is not None:
            data["question[incorrect_comments]"] = question_incorrect_comments

        # OPTIONAL - question[neutral_comments]
        """The comment to display regardless of how the student answered."""
        if question_neutral_comments is not None:
            data["question[neutral_comments]"] = question_neutral_comments

        # OPTIONAL - question[text_after_answers]
        """no description"""
        if question_text_after_answers is not None:
            data["question[text_after_answers]"] = question_text_after_answers

        # OPTIONAL - question[answers]
        """no description"""
        if question_answers is not None:
            data["question[answers]"] = question_answers

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes/{quiz_id}/questions with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/questions".format(**path), data=data, params=params, single_item=True)