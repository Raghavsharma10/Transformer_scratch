def user_id(self):
        """
        Return the ID of the current request's user
        """
        # This needs flask-login to be installed
        if not has_flask_login:
            return

        # and the actual login manager installed
        if not hasattr(current_app, 'login_manager'):
            return

        # fail if no current_user was attached to the request context
        try:
            is_authenticated = current_user.is_authenticated
        except AttributeError:
            return

        # because is_authenticated could be a callable, call it
        if callable(is_authenticated):
            is_authenticated = is_authenticated()

        # and fail if the user isn't authenticated
        if not is_authenticated:
            return

        # finally return the user id
        return current_user.get_id()