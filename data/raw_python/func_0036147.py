def assoc_host(self, hostname, env):
        """
        Associate a host with an environment.

        hostname is opaque to Jones.
        Any string which uniquely identifies a host is acceptable.
        """

        dest = self._get_view_path(env)
        self.associations.set(hostname, dest)