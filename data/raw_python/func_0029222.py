def get_authuser_by_name(cls, request):
        """ Get user by username

        Used by Token-based auth. Is added as request method to populate
        `request.user`.
        """
        username = authenticated_userid(request)
        if username:
            return cls.get_item(username=username)