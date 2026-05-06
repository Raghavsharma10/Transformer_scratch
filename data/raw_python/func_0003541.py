def clear_session(self, response):
        """Clear the session.

        This method is invoked when the session is found to be invalid.
        Subclasses can override this method to implement a custom session
        reset.
        """
        session.clear()

        # if flask-login is installed, we try to clear the
        # "remember me" cookie, just in case it is set
        if 'flask_login' in sys.modules:
            remember_cookie = current_app.config.get('REMEMBER_COOKIE',
                                                     'remember_token')
            response.set_cookie(remember_cookie, '', expires=0, max_age=0)