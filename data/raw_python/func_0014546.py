def check(self, hostname, key):
        """
        Return True if the given key is associated with the given hostname
        in this dictionary.

        @param hostname: hostname (or IP) of the SSH server
        @type hostname: str
        @param key: the key to check
        @type key: L{PKey}
        @return: C{True} if the key is associated with the hostname; C{False}
            if not
        @rtype: bool
        """
        k = self.lookup(hostname)
        if k is None:
            return False
        host_key = k.get(key.get_name(), None)
        if host_key is None:
            return False
        return str(host_key) == str(key)