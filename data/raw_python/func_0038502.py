def delete_topic_groups(self, group_id, topic_id):
        """
        Delete a topic.

        Deletes the discussion topic. This will also delete the assignment, if it's
        an assignment discussion.
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

        self.logger.debug("DELETE /api/v1/groups/{group_id}/discussion_topics/{topic_id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("DELETE", "/api/v1/groups/{group_id}/discussion_topics/{topic_id}".format(**path), data=data, params=params, no_data=True)