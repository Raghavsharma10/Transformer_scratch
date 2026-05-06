def github_authenticated(cls, func):
        """Does user authentication, creates SSH keys if needed and injects "_user" attribute
        into class/object bound to the decorated function.
        Don't call any other methods of this class manually, this should be everything you need.
        """
        def inner(func_cls, *args, **kwargs):
            if not cls._gh_module:
                logger.warning('PyGithub not installed, skipping Github auth procedures.')
            elif not func_cls._user:
                # authenticate user, possibly also creating authentication for future use
                login = kwargs['login'].encode(utils.defenc) if not six.PY3 else kwargs['login']
                func_cls._user = cls._get_github_user(login, kwargs['ui'])
                if func_cls._user is None:
                    msg = 'Github authentication failed, skipping Github command.'
                    logger.warning(msg)
                    return (False, msg)
                # create an ssh key for pushing if we don't have one
                if not cls._github_ssh_key_exists():
                    cls._github_create_ssh_key()
                # next, create ~/.ssh/config entry for the key, if system username != GH login
                if cls._ssh_key_needs_config_entry():
                    cls._create_ssh_config_entry()
            return func(func_cls, *args, **kwargs)

        return inner