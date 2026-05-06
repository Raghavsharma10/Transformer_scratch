def save(self):
        """
        Save profile settings into user profile directory
        """
        config = self.profiledir + '/config'
        if not isdir(self.profiledir):
            makedirs(self.profiledir)

        cp = SafeConfigParser()

        cp.add_section('ssh')
        cp.set('ssh', 'private_key', self.ssh_private_key)
        cp.set('ssh', 'public_key', self.ssh_public_key)

        with open(config, 'w') as cfile:
            cp.write(cfile)