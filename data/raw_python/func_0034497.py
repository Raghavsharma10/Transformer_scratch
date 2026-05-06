def install_key_data(self, keydata, target):
        """Install the key data into the open file."""

        target.seek(0)
        contents = target.read()
        ssh_opts = 'no-port-forwarding'
        if keydata in contents:
            raise InstallError('key data already in file - refusing '
                               'to double-install.\n')
        command = '%s --run' % self.authprogs_binary
        if self.logfile:
            command += ' --logfile=%s' % self.logfile
        if self.keyname:
            command += ' --keyname=%s' % self.keyname

        target.write('command="%(command)s",%(ssh_opts)s %(keydata)s\n' %
                     {'command': command,
                      'keydata': keydata,
                      'ssh_opts': ssh_opts})