def add_message(self, id, body, attachment_ids=None, included_messages=None, media_comment_id=None, media_comment_type=None, recipients=None, user_note=None):
        """
        Add a message.

        Add a message to an existing conversation. Response is similar to the
        GET/show action, except that only includes the
        latest message (i.e. what we just sent)
        
        An array of user ids. Defaults to all of the current conversation
        recipients. To explicitly send a message to no other recipients,
        this array should consist of the logged-in user id.
        
        An array of message ids from this conversation to send to recipients
        of the new message. Recipients who already had a copy of included
        messages will not be affected.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # REQUIRED - body
        """The message to be sent."""
        data["body"] = body

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
        """Type of the associated media file."""
        if media_comment_type is not None:
            self._validate_enum(media_comment_type, ["audio", "video"])
            data["media_comment_type"] = media_comment_type

        # OPTIONAL - recipients
        """no description"""
        if recipients is not None:
            data["recipients"] = recipients

        # OPTIONAL - included_messages
        """no description"""
        if included_messages is not None:
            data["included_messages"] = included_messages

        # OPTIONAL - user_note
        """Will add a faculty journal entry for each recipient as long as the user
        making the api call has permission, the recipient is a student and
        faculty journals are enabled in the account."""
        if user_note is not None:
            data["user_note"] = user_note

        self.logger.debug("POST /api/v1/conversations/{id}/add_message with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/conversations/{id}/add_message".format(**path), data=data, params=params, no_data=True)