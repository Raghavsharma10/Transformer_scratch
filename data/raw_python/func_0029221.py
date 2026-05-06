def get_authuser_by_userid(cls, request):
        """ Get user by ID.

        Used by Ticket-based auth. Is added as request method to populate
        `request.user`.
        """
        userid = authenticated_userid(request)
        if userid:
            cache_request_user(cls, request, userid)
            return request._user