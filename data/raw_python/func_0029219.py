def get_groups_by_userid(cls, userid, request):
        """ Return group identifiers of user with id :userid:

        Used by Ticket-based auth as `callback` kwarg.
        """
        try:
            cache_request_user(cls, request, userid)
        except Exception as ex:
            log.error(str(ex))
            forget(request)
        else:
            if request._user:
                return ['g:%s' % g for g in request._user.groups]