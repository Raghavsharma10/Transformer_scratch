def unflagging_question(self, id, attempt, validation_token, quiz_submission_id, access_code=None):
        """
        Unflagging a question.

        Remove the flag that you previously set on a quiz question after you've
        returned to it.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - quiz_submission_id
        """ID"""
        path["quiz_submission_id"] = quiz_submission_id

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - attempt
        """The attempt number of the quiz submission being taken. Note that this
        must be the latest attempt index, as questions for earlier attempts can
        not be modified."""
        data["attempt"] = attempt

        # REQUIRED - validation_token
        """The unique validation token you received when the Quiz Submission was
        created."""
        data["validation_token"] = validation_token

        # OPTIONAL - access_code
        """Access code for the Quiz, if any."""
        if access_code is not None:
            data["access_code"] = access_code

        self.logger.debug("PUT /api/v1/quiz_submissions/{quiz_submission_id}/questions/{id}/unflag with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/quiz_submissions/{quiz_submission_id}/questions/{id}/unflag".format(**path), data=data, params=params, no_data=True)