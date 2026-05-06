def _github_store_authorization(cls, user, auth):
        """Store an authorization token for the given GitHub user in the git
           global config file.
        """
        ClHelper.run_command("git config --global github.token.{login} {token}".format(
            login=user.login, token=auth.token), log_secret=True)
        ClHelper.run_command("git config --global github.user.{login} {login}".format(
            login=user.login))