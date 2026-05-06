def delete_message(self, id, remove):
        """
        Delete a message.

        Delete messages from this conversation. Note that this only affects this
        user's view of the conversation. If all messages are deleted, the
        conversation will be as well (equivalent to DELETE)
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - remove
        """Array of message ids to be deleted"""
        data["remove"] = remove

        self.logger.debug("POST /api/v1/conversations/{id}/remove_messages with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/conversations/{id}/remove_messages".format(**path), data=data, params=params, no_data=True)