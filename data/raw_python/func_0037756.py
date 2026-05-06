def reorder_question_groups(self, id, quiz_id, order_id, course_id, order_type=None):
        """
        Reorder question groups.

        Change the order of the quiz questions within the group
        
        <b>204 No Content<b> response code is returned if the reorder was successful.
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

        # REQUIRED - order[id]
        """The associated item's unique identifier"""
        data["order[id]"] = order_id

        # OPTIONAL - order[type]
        """The type of item is always 'question' for a group"""
        if order_type is not None:
            self._validate_enum(order_type, ["question"])
            data["order[type]"] = order_type

        self.logger.debug("POST /api/v1/courses/{course_id}/quizzes/{quiz_id}/groups/{id}/reorder with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/quizzes/{quiz_id}/groups/{id}/reorder".format(**path), data=data, params=params, no_data=True)