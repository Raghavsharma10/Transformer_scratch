def get_groups_by_token(cls, username, token, request):
        """ Get user's groups if user with :username: exists and their api key
        token equals :token:

        Used by Token-based authentication as `check` kwarg.
        """
        try:
            user = cls.get_item(username=username)
        except Exception as ex:
            log.error(str(ex))
            forget(request)
            return
        else:
            if user and user.api_key.token == token:
                return ['g:%s' % g for g in user.groups]