def get_single_conversation(self, id, auto_mark_as_read=None, filter=None, filter_mode=None, interleave_submissions=None, scope=None):
        """
        Get a single conversation.

        Returns information for a single conversation for the current user. Response includes all
        fields that are present in the list/index action as well as messages
        and extended participant information.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - interleave_submissions
        """(Obsolete) Submissions are no
        longer linked to conversations. This parameter is ignored."""
        if interleave_submissions is not None:
            params["interleave_submissions"] = interleave_submissions

        # OPTIONAL - scope
        """Used when generating "visible" in the API response. See the explanation
        under the {api:ConversationsController#index index API action}"""
        if scope is not None:
            self._validate_enum(scope, ["unread", "starred", "archived"])
            params["scope"] = scope

        # OPTIONAL - filter
        """Used when generating "visible" in the API response. See the explanation
        under the {api:ConversationsController#index index API action}"""
        if filter is not None:
            params["filter"] = filter

        # OPTIONAL - filter_mode
        """Used when generating "visible" in the API response. See the explanation
        under the {api:ConversationsController#index index API action}"""
        if filter_mode is not None:
            self._validate_enum(filter_mode, ["and", "or", "default or"])
            params["filter_mode"] = filter_mode

        # OPTIONAL - auto_mark_as_read
        """Default true. If true, unread
        conversations will be automatically marked as read. This will default
        to false in a future API release, so clients should explicitly send
        true if that is the desired behavior."""
        if auto_mark_as_read is not None:
            params["auto_mark_as_read"] = auto_mark_as_read

        self.logger.debug("GET /api/v1/conversations/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/conversations/{id}".format(**path), data=data, params=params, no_data=True)