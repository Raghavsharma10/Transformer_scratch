def authenticate_by_password(cls, params):
        """ Authenticate user with login and password from :params:

        Used both by Token and Ticket-based auths (called from views).
        """
        def verify_password(user, password):
            return crypt.check(user.password, password)

        success = False
        user = None
        login = params['login'].lower().strip()
        key = 'email' if '@' in login else 'username'
        try:
            user = cls.get_item(**{key: login})
        except Exception as ex:
            log.error(str(ex))

        if user:
            password = params.get('password', None)
            success = (password and verify_password(user, password))
        return success, user