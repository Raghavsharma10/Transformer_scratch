def requires_auth(self, roles=None):
        """
        Used to impose auth constraints on requests which require a logged in user with particular roles.

        :param list[string] roles:
            A list of :class:`string` representing roles the logged in user must have to perform this action. The user
            and password are passed in each request in the authorization header obtained from request.authorization,
            the user and password are checked against the user database and roles obtained. The user must match an
            existing user (including the password, obviously) and must have every role specified in this parameter.
        :return:
            The result of the wrapped function if everything is okay, or a flask.abort(403) error code if authentication
            fails, either because the user isn't properly authenticated or because the user doesn't have the required
            role or roles.
        """

        def requires_auth_inner(f):
            @wraps(f)
            def decorated(*args, **kwargs):
                auth = request.authorization
                if not auth:
                    return MeteorApp.authentication_failure(message='No authorization header supplied')
                user_id = auth.username
                password = auth.password
                try:
                    db = self.get_db()
                    user = db.get_user(user_id=user_id, password=password)
                    if user is None:
                        return MeteorApp.authentication_failure(message='Username and / or password incorrect')
                    if roles is not None:
                        for role in roles:
                            if not user.has_role(role):
                                return MeteorApp.authentication_failure(message='Missing role {0}'.format(role))
                    g.user = user
                    db.close_db()
                except ValueError:
                    return MeteorApp.authentication_failure(message='Unrecognized role encountered')
                return f(*args, **kwargs)

            return decorated

        return requires_auth_inner