def create_account(cls, params):
        """ Create auth user instance with data from :params:.

        Used by both Token and Ticket-based auths to register a user (
        called from views).
        """
        user_params = dictset(params).subset(
            ['username', 'email', 'password'])
        try:
            return cls.get_or_create(
                email=user_params['email'],
                defaults=user_params)
        except JHTTPBadRequest:
            raise JHTTPBadRequest('Failed to create account.')