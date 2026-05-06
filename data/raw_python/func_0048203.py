def check_perms(self, perms='0600,0400'):
        """Check and enforce the permissions of the config file.

        Enforce permission on a provided configuration file. This will check
        and see if the permission are set based on the permission octet as
        set in the ``perms`` value. ``perms`` is a comma separated list
        of acceptable perms in octal form. Defaults permissions to, 0600 and
        0400.

        :param perms: ``str``
        """
        confpath = os.path.realpath(self.config_file)
        mode = stat.S_IMODE(os.stat(confpath).st_mode)
        if not any([mode == int(i, 8) for i in perms.split(',')]):
            msg = (
                'To use a configuration file the permissions'
                ' need to be any of the following "%s"' % perms
            )
            self.log.fatal(msg)
            raise SystemExit(msg)
        else:
            self.log.info(
                'Configuration file [ %s ] has been loaded',
                self.config_file
            )
            return True