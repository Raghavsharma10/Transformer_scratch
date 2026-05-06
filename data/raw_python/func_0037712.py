def create_conversation(self, body, recipients, attachment_ids=None, context_code=None, filter=None, filter_mode=None, group_conversation=None, media_comment_id=None, media_comment_type=None, mode=None, scope=None, subject=None, user_note=None):
        """
        Create a conversation.

        Create a new conversation with one or more recipients. If there is already
        an existing private conversation with the given recipients, it will be
        reused.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - recipients
        """An array of recipient ids. These may be user ids or course/group ids
        prefixed with "course_" or "group_" respectively, e.g.
        recipients[]=1&recipients[]=2&recipients[]=course_3"""
        data["recipients"] = recipients

        # OPTIONAL - subject
        """The subject of the conversation. This is ignored when reusing a
        conversation. Maximum length is 255 characters."""
        if subject is not None:
            data["subject"] = subject

        # REQUIRED - body
        """The message to be sent"""
        data["body"] = body

        # OPTIONAL - group_conversation
        """Defaults to false. If true, this will be a group conversation (i.e. all
        recipients may see all messages and replies). If false, individual private
        conversations will be started with each recipient. Must be set false if the
        number of recipients is over the set maximum (default is 100)."""
        if group_conversation is not None:
            data["group_conversation"] = group_conversation

        # OPTIONAL - attachment_ids
        """An array of attachments ids. These must be files that have been previously
        uploaded to the sender's "conversation attachments" folder."""
        if attachment_ids is not None:
            data["attachment_ids"] = attachment_ids

        # OPTIONAL - media_comment_id
        """Media comment id of an audio of video file to be associated with this
        message."""
        if media_comment_id is not None:
            data["media_comment_id"] = media_comment_id

        # OPTIONAL - media_comment_type
        """Type of the associated media file"""
        if media_comment_type is not None:
            self._validate_enum(media_comment_type, ["audio", "video"])
            data["media_comment_type"] = media_comment_type

        # OPTIONAL - user_note
        """Will add a faculty journal entry for each recipient as long as the user
        making the api call has permission, the recipient is a student and
        faculty journals are enabled in the account."""
        if user_note is not None:
            data["user_note"] = user_note

        # OPTIONAL - mode
        """Determines whether the messages will be created/sent synchronously or
        asynchronously. Defaults to sync, and this option is ignored if this is a
        group conversation or there is just one recipient (i.e. it must be a bulk
        private message). When sent async, the response will be an empty array
        (batch status can be queried via the {api:ConversationsController#batches batches API})"""
        if mode is not None:
            self._validate_enum(mode, ["sync", "async"])
            data["mode"] = mode

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

        # OPTIONAL - context_code
        """The course or group that is the context for this conversation. Same format
        as courses or groups in the recipients argument."""
        if context_code is not None:
            data["context_code"] = context_code

        self.logger.debug("POST /api/v1/conversations with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/conversations".format(**path), data=data, params=params, no_data=True)