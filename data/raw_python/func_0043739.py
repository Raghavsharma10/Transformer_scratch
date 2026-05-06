def setup_salt_ssh(self):
        """
        Setup `salt-ssh`
        """
        self.copy_salt_and_pillar()
        self.create_roster_file()
        self.salt_ssh_create_dirs()
        self.salt_ssh_create_master_file()