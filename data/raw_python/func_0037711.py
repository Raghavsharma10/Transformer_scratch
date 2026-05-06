def list_conversations(self, filter=None, filter_mode=None, include=None, include_all_conversation_ids=None, interleave_submissions=None, scope=None):
        """
        List conversations.

        Returns the list of conversations for the current user, most recent ones first.
        """
        path = {}
        data = {}
        params = {}

        # OPTIONAL - scope
        """When set, only return conversations of the specified type. For example,
        set to "unread" to return only conversations that haven't been read.
        The default behavior is to return all non-archived conversations (i.e.
        read and unread)."""
        if scope is not None:
            self._validate_enum(scope, ["unread", "starred", "archived"])
            params["scope"] = scope

        # OPTIONAL - filter
        """When set, only return conversations for the specified courses, groups
        or users. The id should be prefixed with its type, e.g. "user_123" or
        "course_456". Can be an array (by setting "filter[]") or single value
        (by setting "filter")"""
        if filter is not None:
            params["filter"] = filter

        # OPTIONAL - filter_mode
        """When filter[] contains multiple filters, combine them with this mode,
        filtering conversations that at have at least all of the contexts ("and")
        or at least one of the contexts ("or")"""
        if filter_mode is not None:
            self._validate_enum(filter_mode, ["and", "or", "default or"])
            params["filter_mode"] = filter_mode

        # OPTIONAL - interleave_submissions
        """(Obsolete) Submissions are no
        longer linked to conversations. This parameter is ignored."""
        if interleave_submissions is not None:
            params["interleave_submissions"] = interleave_submissions

        # OPTIONAL - include_all_conversation_ids
        """Default is false. If true,
        the top-level element of the response will be an object rather than
        an array, and will have the keys "conversations" which will contain the
        paged conversation data, and "conversation_ids" which will contain the
        ids of all conversations under this scope/filter in the same order."""
        if include_all_conversation_ids is not None:
            params["include_all_conversation_ids"] = include_all_conversation_ids

        # OPTIONAL - include
        """"participant_avatars":: Optionally include an "avatar_url" key for each user participanting in the conversation"""
        if include is not None:
            self._validate_enum(include, ["participant_avatars"])
            params["include"] = include

        self.logger.debug("GET /api/v1/conversations with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("GET", "/api/v1/conversations".format(**path), data=data, params=params, all_pages=True)