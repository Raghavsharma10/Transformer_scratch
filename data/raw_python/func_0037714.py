def edit_conversation(self, id, conversation_starred=None, conversation_subscribed=None, conversation_workflow_state=None, filter=None, filter_mode=None, scope=None):
        """
        Edit a conversation.

        Updates attributes for a single conversation.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - conversation[workflow_state]
        """Change the state of this conversation"""
        if conversation_workflow_state is not None:
            self._validate_enum(conversation_workflow_state, ["read", "unread", "archived"])
            data["conversation[workflow_state]"] = conversation_workflow_state

        # OPTIONAL - conversation[subscribed]
        """Toggle the current user's subscription to the conversation (only valid for
        group conversations). If unsubscribed, the user will still have access to
        the latest messages, but the conversation won't be automatically flagged
        as unread, nor will it jump to the top of the inbox."""
        if conversation_subscribed is not None:
            data["conversation[subscribed]"] = conversation_subscribed

        # OPTIONAL - conversation[starred]
        """Toggle the starred state of the current user's view of the conversation."""
        if conversation_starred is not None:
            data["conversation[starred]"] = conversation_starred

        # OPTIONAL - scope
        """Used when generating "visible" in the API response. See the explanation
        under the {api:ConversationsController#index index API action}"""
        if scope is not None:
            self._validate_enum(scope, ["unread", "starred", "archived"])
            data["scope"] = scope

        # OPTIONAL - filter
        """Used when generating "visible" in the API response. See the explanation
        under the {api:ConversationsController#index index API action}"""
        if filter is not None:
            data["filter"] = filter

        # OPTIONAL - filter_mode
        """Used when generating "visible" in the API response. See the explanation
        under the {api:ConversationsController#index index API action}"""
        if filter_mode is not None:
            self._validate_enum(filter_mode, ["and", "or", "default or"])
            data["filter_mode"] = filter_mode

        self.logger.debug("PUT /api/v1/conversations/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/conversations/{id}".format(**path), data=data, params=params, no_data=True)