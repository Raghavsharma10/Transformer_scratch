def get_current_user(self):
        """
        Override get_current_user for Google AppEngine
        Checks for oauth capable request first, if this fails fall back to standard users API
        """
        from google.appengine.api import users

        if _IS_DEVELOPMENT_SERVER:
            return users.get_current_user()
        else:
            from google.appengine.api import oauth

            try:
                user = oauth.get_current_user()
            except oauth.OAuthRequestError:
                user = users.get_current_user()
            return user