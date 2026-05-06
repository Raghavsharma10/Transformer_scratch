def batch_update_conversations(self, event, conversation_ids):
        """
        Batch update conversations.

        Perform a change on a set of conversations. Operates asynchronously; use the {api:ProgressController#show progress endpoint}
        to query the status of an operation.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - conversation_ids
        """List of conversations to update. Limited to 500 conversations."""
        data["conversation_ids"] = conversation_ids

        # REQUIRED - event
        """The action to take on each conversation."""
        self._validate_enum(event, ["mark_as_read", "mark_as_unread", "star", "unstar", "archive", "destroy"])
        data["event"] = event

        self.logger.debug("PUT /api/v1/conversations with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/conversations".format(**path), data=data, params=params, single_item=True)