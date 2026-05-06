def post_reply_groups(self, group_id, topic_id, entry_id, attachment=None, message=None):
        """
        Post a reply.

        Add a reply to an entry in a discussion topic. Returns a json
        representation of the created reply (see documentation for 'replies'
        method) on success.
        
        May require (depending on the topic) that the user has posted in the topic.
        If it is required, and the user has not posted, will respond with a 403
        Forbidden status and the body 'require_initial_post'.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - group_id
        """ID"""
        path["group_id"] = group_id

        # REQUIRED - PATH - topic_id
        """ID"""
        path["topic_id"] = topic_id

        # REQUIRED - PATH - entry_id
        """ID"""
        path["entry_id"] = entry_id

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

        self.logger.debug("POST /api/v1/groups/{group_id}/discussion_topics/{topic_id}/entries/{entry_id}/replies with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/groups/{group_id}/discussion_topics/{topic_id}/entries/{entry_id}/replies".format(**path), data=data, params=params, no_data=True)