def load_config(self, name=None, path=None, home=False, ext='ini'):
        """Return the full path to a configuration file.

        This will look for configuration files in the ``full_path``, in
        the users home folder, or in a config directory in "/etc".  If you set
        `home=True`, the system will look for a "hidden" configuration file in
        the executing users $HOME folder.

        The precedence is as follows:
            full_path/name.ini
            /home/$USER/.name.ini
            /etc/name/name.ini

        :param name: ``str``
        :param path: ``str``
        :param home: ``bol``
        :param ext: ``str``
        :return: ``str``
        """

        opj = os.path.join

        self.name = name
        self.filename = '%s.%s' % (self.name, ext)

        checks = []
        if path is not None:
            checks.append(opj(path.rstrip(os.sep), self.filename))
        elif home is True:
            checks.append(opj(os.getenv('HOME'), '%s' % self.filename))
        else:
            checks.append(opj('/etc', self.name, self.filename))

        for check in checks:
            if os.path.exists(check):
                self.config_file = check
                break
        else:
            self._find_config(name)