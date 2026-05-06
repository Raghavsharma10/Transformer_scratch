def _github_create_simple_authorization(cls):
        """Create a GitHub authorization for the given user in case they don't
           already have one.
        """
        try:
            auth = None
            for a in cls._user.get_authorizations():
                if a.note == 'DevAssistant':
                    auth = a
            if not auth:
                auth = cls._user.create_authorization(
                    scopes=['repo', 'user', 'admin:public_key'],
                    note="DevAssistant")
                cls._github_store_authorization(cls._user, auth)
        except cls._gh_exceptions.GithubException as e:
            logger.warning('Creating authorization failed: {0}'.format(e))