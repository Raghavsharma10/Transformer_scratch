def _choose_port(self):
        """
        Return a port number from 5000-5999 based on the environment name
        to be used as a default when the user hasn't selected one.
        """
        # instead of random let's base it on the name chosen (and the site name)
        return 5000 + unpack('Q',
                             sha((self.name + self.site_name)
                             .decode('ascii')).digest()[:8])[0] % 1000