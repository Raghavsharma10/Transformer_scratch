def init_host(self):
        """
            Initial host
        """
        env.host_string = self.host_string
        env.user = self.host_user
        env.password = self.host_passwd
        env.key_filename = self.host_keyfile