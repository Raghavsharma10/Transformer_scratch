def list(self, request, *args, **kwargs):
        """
        User list is available to all authenticated users. To get a list,
        issue authenticated **GET** request against */api/users/*.

        User list supports several filters. All filters are set in HTTP query section.
        Field filters are listed below. All of the filters apart from ?organization are
        using case insensitive partial matching.

        Several custom filters are supported:

        - ?current - filters out user making a request. Useful for getting information about a currently logged in user.
        - ?civil_number=XXX - filters out users with a specified civil number
        - ?is_active=True|False - show only active (non-active) users

        The user can be created either through automated process on login with SAML token, or through a REST call by a user
        with staff privilege.

        Example of a creation request is below.

        .. code-block:: http

            POST /api/users/ HTTP/1.1
            Content-Type: application/json
            Accept: application/json
            Authorization: Token c84d653b9ec92c6cbac41c706593e66f567a7fa4
            Host: example.com

            {
                "username": "sample-user",
                "full_name": "full name",
                "native_name": "taisnimi",
                "job_title": "senior cleaning manager",
                "email": "example@example.com",
                "civil_number": "12121212",
                "phone_number": "",
                "description": "",
                "organization": "",
            }

        NB! Username field is case-insensitive. So "John" and "john" will be treated as the same user.
        """
        return super(UserViewSet, self).list(request, *args, **kwargs)