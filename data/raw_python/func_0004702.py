def oauth_only(function):
    """Decorator to restrict some GitHubTools methods to run only with OAuth"""

    def check_for_oauth(self, *args, **kwargs):
        """
        Returns False if GitHubTools instance is not authenticated, or return
        the decorated fucntion if it is.
        """
        if not self.is_authenticated:
            self.oops("To use putgist you have to set your GETGIST_TOKEN")
            self.oops("(see `putgist --help` for details)")
            return False
        return function(self, *args, **kwargs)

    return check_for_oauth