def salt_ssh_create_dirs(self):
        """
        Creates the `salt-ssh` required directory structure
        """
        logger.debug('Creating salt-ssh dirs into: %s', self.settings_dir)
        utils.create_dir(os.path.join(self.settings_dir, 'salt'))
        utils.create_dir(os.path.join(self.settings_dir, 'pillar'))
        utils.create_dir(os.path.join(self.settings_dir, 'etc', 'salt'))
        utils.create_dir(os.path.join(self.settings_dir, 'var', 'cache', 'salt'))
        utils.create_dir(os.path.join(self.settings_dir, 'var', 'log', 'salt'))