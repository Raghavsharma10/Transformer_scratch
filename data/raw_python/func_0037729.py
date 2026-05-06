def self_register_user(self, user_name, account_id, user_terms_of_use, pseudonym_unique_id, communication_channel_address=None, communication_channel_type=None, user_birthdate=None, user_locale=None, user_short_name=None, user_sortable_name=None, user_time_zone=None):
        """
        Self register a user.

        Self register and return a new user and pseudonym for an account.
        
        If self-registration is enabled on the account, you can use this
        endpoint to self register new users.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # REQUIRED - user[name]
        """The full name of the user. This name will be used by teacher for grading."""
        data["user[name]"] = user_name

        # OPTIONAL - user[short_name]
        """User's name as it will be displayed in discussions, messages, and comments."""
        if user_short_name is not None:
            data["user[short_name]"] = user_short_name

        # OPTIONAL - user[sortable_name]
        """User's name as used to sort alphabetically in lists."""
        if user_sortable_name is not None:
            data["user[sortable_name]"] = user_sortable_name

        # OPTIONAL - user[time_zone]
        """The time zone for the user. Allowed time zones are
        {http://www.iana.org/time-zones IANA time zones} or friendlier
        {http://api.rubyonrails.org/classes/ActiveSupport/TimeZone.html Ruby on Rails time zones}."""
        if user_time_zone is not None:
            data["user[time_zone]"] = user_time_zone

        # OPTIONAL - user[locale]
        """The user's preferred language, from the list of languages Canvas supports.
        This is in RFC-5646 format."""
        if user_locale is not None:
            data["user[locale]"] = user_locale

        # OPTIONAL - user[birthdate]
        """The user's birth date."""
        if user_birthdate is not None:
            data["user[birthdate]"] = user_birthdate

        # REQUIRED - user[terms_of_use]
        """Whether the user accepts the terms of use."""
        data["user[terms_of_use]"] = user_terms_of_use

        # REQUIRED - pseudonym[unique_id]
        """User's login ID. Must be a valid email address."""
        data["pseudonym[unique_id]"] = pseudonym_unique_id

        # OPTIONAL - communication_channel[type]
        """The communication channel type, e.g. 'email' or 'sms'."""
        if communication_channel_type is not None:
            data["communication_channel[type]"] = communication_channel_type

        # OPTIONAL - communication_channel[address]
        """The communication channel address, e.g. the user's email address."""
        if communication_channel_address is not None:
            data["communication_channel[address]"] = communication_channel_address

        self.logger.debug("POST /api/v1/accounts/{account_id}/self_registration with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/self_registration".format(**path), data=data, params=params, single_item=True)