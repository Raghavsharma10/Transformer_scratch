def edit_user(self, id, user_avatar_token=None, user_avatar_url=None, user_email=None, user_locale=None, user_name=None, user_short_name=None, user_sortable_name=None, user_time_zone=None):
        """
        Edit a user.

        Modify an existing user. To modify a user's login, see the documentation for logins.
        """
        path = {}
        data = {}
        params = {}

        # REQUIRED - PATH - id
        """ID"""
        path["id"] = id

        # OPTIONAL - user[name]
        """The full name of the user. This name will be used by teacher for grading."""
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

        # OPTIONAL - user[email]
        """The default email address of the user."""
        if user_email is not None:
            data["user[email]"] = user_email

        # OPTIONAL - user[locale]
        """The user's preferred language, from the list of languages Canvas supports.
        This is in RFC-5646 format."""
        if user_locale is not None:
            data["user[locale]"] = user_locale

        # OPTIONAL - user[avatar][token]
        """A unique representation of the avatar record to assign as the user's
        current avatar. This token can be obtained from the user avatars endpoint.
        This supersedes the user [avatar] [url] argument, and if both are included
        the url will be ignored. Note: this is an internal representation and is
        subject to change without notice. It should be consumed with this api
        endpoint and used in the user update endpoint, and should not be
        constructed by the client."""
        if user_avatar_token is not None:
            data["user[avatar][token]"] = user_avatar_token

        # OPTIONAL - user[avatar][url]
        """To set the user's avatar to point to an external url, do not include a
        token and instead pass the url here. Warning: For maximum compatibility,
        please use 128 px square images."""
        if user_avatar_url is not None:
            data["user[avatar][url]"] = user_avatar_url

        self.logger.debug("PUT /api/v1/users/{id} with query params: {params} and form data: {data}".format(params=params, data=data, **path))
        return self.generic_request("PUT", "/api/v1/users/{id}".format(**path), data=data, params=params, single_item=True)