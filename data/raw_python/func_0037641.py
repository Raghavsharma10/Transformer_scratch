def create_communication_channel(self, user_id, communication_channel_type, communication_channel_address, communication_channel_token=None, skip_confirmation=None):
        """
        Create a communication channel.

        Creates a new communication channel for the specified user.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - user_id
        """ID"""
        path["user_id"] = user_id

        # REQUIRED - communication_channel[address]
        """An email address or SMS number. Not required for "push" type channels."""
        data["communication_channel[address]"] = communication_channel_address

        # REQUIRED - communication_channel[type]
        """The type of communication channel.
        
        In order to enable push notification support, the server must be
        properly configured (via sns.yml) to communicate with Amazon
        Simple Notification Services, and the developer key used to create
        the access token from this request must have an SNS ARN configured on
        it."""
        self._validate_enum(communication_channel_type, ["email", "sms", "push"])
        data["communication_channel[type]"] = communication_channel_type

        # OPTIONAL - communication_channel[token]
        """A registration id, device token, or equivalent token given to an app when
        registering with a push notification provider. Only valid for "push" type channels."""
        if communication_channel_token is not None:
            data["communication_channel[token]"] = communication_channel_token

        # OPTIONAL - skip_confirmation
        """Only valid for site admins and account admins making requests; If true, the channel is
        automatically validated and no confirmation email or SMS is sent.
        Otherwise, the user must respond to a confirmation message to confirm the
        channel."""
        if skip_confirmation is not None:
            data["skip_confirmation"] = skip_confirmation

        self.logger.debug("POST /api/v1/users/{user_id}/communication_channels with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/users/{user_id}/communication_channels".format(**path), data=data, params=params, single_item=True)