def _github_create_ssh_key(cls):
        """Creates a local ssh key, if it doesn't exist already, and uploads it to Github."""
        try:
            login = cls._user.login
            pkey_path = '{home}/.ssh/{keyname}'.format(
                home=os.path.expanduser('~'),
                keyname=settings.GITHUB_SSH_KEYNAME.format(login=login))
            # generate ssh key only if it doesn't exist
            if not os.path.exists(pkey_path):
                ClHelper.run_command('ssh-keygen -t rsa -f {pkey_path}\
                                     -N \"\" -C \"DevAssistant\"'.
                                     format(pkey_path=pkey_path))
            try:
                ClHelper.run_command('ssh-add {pkey_path}'.format(pkey_path=pkey_path))
            except exceptions.ClException:
                # ssh agent might not be running
                env = cls._start_ssh_agent()
                ClHelper.run_command('ssh-add {pkey_path}'.format(pkey_path=pkey_path), env=env)
            public_key = ClHelper.run_command('cat {pkey_path}.pub'.format(pkey_path=pkey_path))
            cls._user.create_key("DevAssistant", public_key)
        except exceptions.ClException as e:
            msg = 'Couldn\'t create a new ssh key: {0}'.format(e)
            raise exceptions.CommandException(msg)