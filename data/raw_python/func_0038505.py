def post_entry_courses(self, topic_id, course_id, attachment=None, message=None):
        """
        Post an entry.

        Create a new entry in a discussion topic. Returns a json representation of
        the created entry (see documentation for 'entries' method) on success.
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

        # OPTIONAL - message
        """The body of the entry."""
        if message is not None:
            data["message"] = message

        # OPTIONAL - attachment
        """a multipart/form-data form-field-style
        attachment. Attachments larger than 1 kilobyte are subject to quota
        restrictions."""
        if attachment is not None:
            data["attachment"] = attachment

        self.logger.debug("POST /api/v1/courses/{course_id}/discussion_topics/{topic_id}/entries with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/courses/{course_id}/discussion_topics/{topic_id}/entries".format(**path), data=data, params=params, no_data=True)