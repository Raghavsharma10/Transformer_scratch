def reorder_pinned_topics_courses(self, order, course_id):
        """
        Reorder pinned topics.

        Puts the pinned discussion topics in the specified order.
        All pinned topics should be included.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - order
        """The ids of the pinned discussion topics in the desired order.
        (For example, "order=104,102,103".)"""
        data["order"] = order

        self.logger.debug("POST /api/v1/courses/{course_id}/discussion_topics/reorder with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/discussion_topics/reorder".format(**path), data=data, params=params, no_data=True)