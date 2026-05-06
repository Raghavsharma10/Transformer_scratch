def create_user(self, account_id, pseudonym_unique_id, communication_channel_address=None, communication_channel_confirmation_url=None, communication_channel_skip_confirmation=None, communication_channel_type=None, enable_sis_reactivation=None, force_validations=None, pseudonym_authentication_provider_id=None, pseudonym_force_self_registration=None, pseudonym_integration_id=None, pseudonym_password=None, pseudonym_send_confirmation=None, pseudonym_sis_user_id=None, user_birthdate=None, user_locale=None, user_name=None, user_short_name=None, user_skip_registration=None, user_sortable_name=None, user_terms_of_use=None, user_time_zone=None):
        """
        Create a user.

        Create and return a new user and pseudonym for an account.
        
        If you don't have the "Modify login details for users" permission, but
        self-registration is enabled on the account, you can still use this
        endpoint to register new users. Certain fields will be required, and
        others will be ignored (see below).
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - account_id
        """ID"""
        path["account_id"] = account_id

        # OPTIONAL - user[name]
        """The full name of the user. This name will be used by teacher for grading.
        Required if this is a self-registration."""
        if user_name is not None:
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

        # OPTIONAL - user[terms_of_use]
        """Whether the user accepts the terms of use. Required if this is a
        self-registration and this canvas instance requires users to accept
        the terms (on by default).
        
        If this is true, it will mark the user as having accepted the terms of use."""
        if user_terms_of_use is not None:
            data["user[terms_of_use]"] = user_terms_of_use

        # OPTIONAL - user[skip_registration]
        """Automatically mark the user as registered.
        
        If this is true, it is recommended to set <tt>"pseudonym[send_confirmation]"</tt> to true as well.
        Otherwise, the user will not receive any messages about their account creation.
        
        The users communication channel confirmation can be skipped by setting
        <tt>"communication_channel[skip_confirmation]"</tt> to true as well."""
        if user_skip_registration is not None:
            data["user[skip_registration]"] = user_skip_registration

        # REQUIRED - pseudonym[unique_id]
        """User's login ID. If this is a self-registration, it must be a valid
        email address."""
        data["pseudonym[unique_id]"] = pseudonym_unique_id

        # OPTIONAL - pseudonym[password]
        """User's password. Cannot be set during self-registration."""
        if pseudonym_password is not None:
            data["pseudonym[password]"] = pseudonym_password

        # OPTIONAL - pseudonym[sis_user_id]
        """SIS ID for the user's account. To set this parameter, the caller must be
        able to manage SIS permissions."""
        if pseudonym_sis_user_id is not None:
            data["pseudonym[sis_user_id]"] = pseudonym_sis_user_id

        # OPTIONAL - pseudonym[integration_id]
        """Integration ID for the login. To set this parameter, the caller must be able to
        manage SIS permissions. The Integration ID is a secondary
        identifier useful for more complex SIS integrations."""
        if pseudonym_integration_id is not None:
            data["pseudonym[integration_id]"] = pseudonym_integration_id

        # OPTIONAL - pseudonym[send_confirmation]
        """Send user notification of account creation if true.
        Automatically set to true during self-registration."""
        if pseudonym_send_confirmation is not None:
            data["pseudonym[send_confirmation]"] = pseudonym_send_confirmation

        # OPTIONAL - pseudonym[force_self_registration]
        """Send user a self-registration style email if true.
        Setting it means the users will get a notification asking them
        to "complete the registration process" by clicking it, setting
        a password, and letting them in.  Will only be executed on
        if the user does not need admin approval.
        Defaults to false unless explicitly provided."""
        if pseudonym_force_self_registration is not None:
            data["pseudonym[force_self_registration]"] = pseudonym_force_self_registration

        # OPTIONAL - pseudonym[authentication_provider_id]
        """The authentication provider this login is associated with. Logins
        associated with a specific provider can only be used with that provider.
        Legacy providers (LDAP, CAS, SAML) will search for logins associated with
        them, or unassociated logins. New providers will only search for logins
        explicitly associated with them. This can be the integer ID of the
        provider, or the type of the provider (in which case, it will find the
        first matching provider)."""
        if pseudonym_authentication_provider_id is not None:
            data["pseudonym[authentication_provider_id]"] = pseudonym_authentication_provider_id

        # OPTIONAL - communication_channel[type]
        """The communication channel type, e.g. 'email' or 'sms'."""
        if communication_channel_type is not None:
            data["communication_channel[type]"] = communication_channel_type

        # OPTIONAL - communication_channel[address]
        """The communication channel address, e.g. the user's email address."""
        if communication_channel_address is not None:
            data["communication_channel[address]"] = communication_channel_address

        # OPTIONAL - communication_channel[confirmation_url]
        """Only valid for account admins. If true, returns the new user account
        confirmation URL in the response."""
        if communication_channel_confirmation_url is not None:
            data["communication_channel[confirmation_url]"] = communication_channel_confirmation_url

        # OPTIONAL - communication_channel[skip_confirmation]
        """Only valid for site admins and account admins making requests; If true, the channel is
        automatically validated and no confirmation email or SMS is sent.
        Otherwise, the user must respond to a confirmation message to confirm the
        channel.
        
        If this is true, it is recommended to set <tt>"pseudonym[send_confirmation]"</tt> to true as well.
        Otherwise, the user will not receive any messages about their account creation."""
        if communication_channel_skip_confirmation is not None:
            data["communication_channel[skip_confirmation]"] = communication_channel_skip_confirmation

        # OPTIONAL - force_validations
        """If true, validations are performed on the newly created user (and their associated pseudonym)
        even if the request is made by a privileged user like an admin. When set to false,
        or not included in the request parameters, any newly created users are subject to
        validations unless the request is made by a user with a 'manage_user_logins' right.
        In which case, certain validations such as 'require_acceptance_of_terms' and
        'require_presence_of_name' are not enforced. Use this parameter to return helpful json
        errors while building users with an admin request."""
        if force_validations is not None:
            data["force_validations"] = force_validations

        # OPTIONAL - enable_sis_reactivation
        """When true, will first try to re-activate a deleted user with matching sis_user_id if possible."""
        if enable_sis_reactivation is not None:
            data["enable_sis_reactivation"] = enable_sis_reactivation

        self.logger.debug("POST /api/v1/accounts/{account_id}/users with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("POST", "/api/v1/accounts/{account_id}/users".format(**path), data=data, params=params, single_item=True)