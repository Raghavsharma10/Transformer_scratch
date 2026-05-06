def _github_create_twofactor_authorization(cls, ui):
        """Create an authorization for a GitHub user using two-factor
           authentication. Unlike its non-two-factor counterpart, this method
           does not traverse the available authentications as they are not
           visible until the user logs in.

           Please note: cls._user's attributes are not accessible until the
           authorization is created due to the way (py)github works.
        """
        try:
            try: # This is necessary to trigger sending a 2FA key to the user
                auth = cls._user.create_authorization()
            except cls._gh_exceptions.GithubException:
                onetime_pw = DialogHelper.ask_for_password(ui, prompt='Your one time password:')
                auth = cls._user.create_authorization(scopes=['repo', 'user', 'admin:public_key'],
                                            note="DevAssistant",
                                            onetime_password=onetime_pw)
                cls._user = cls._gh_module.Github(login_or_token=auth.token).get_user()
                logger.debug('Two-factor authorization for user "{0}" created'.format(cls._user.login))
                cls._github_store_authorization(cls._user, auth)
                logger.debug('Two-factor authorization token stored')
        except cls._gh_exceptions.GithubException as e:
            logger.warning('Creating two-factor authorization failed: {0}'.format(e))