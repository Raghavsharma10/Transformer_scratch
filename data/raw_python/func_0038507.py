def rate_entry_courses(self, topic_id, entry_id, course_id, rating=None):
        """
        Rate entry.

        Rate a discussion entry.
        
        On success, the response will be 204 No Content with an empty body.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - course_id
        """ID"""
        path["course_id"] = course_id

        # REQUIRED - PATH - topic_id
        """ID"""
        path["topic_id"] = topic_id

        # REQUIRED - PATH - entry_id
        """ID"""
        path["entry_id"] = entry_id

        # OPTIONAL - rating
        """A rating to set on this entry. Only 0 and 1 are accepted."""
        if rating is not None:
            data["rating"] = rating

        self.logger.debug("POST /api/v1/courses/{course_id}/discussion_topics/{topic_id}/entries/{entry_id}/rating with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/discussion_topics/{topic_id}/entries/{entry_id}/rating".format(**path), data=data, params=params, no_data=True)