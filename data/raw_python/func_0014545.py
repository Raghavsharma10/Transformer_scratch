def lookup(self, hostname):
        """
        Find a hostkey entry for a given hostname or IP.  If no entry is found,
        C{None} is returned.  Otherwise a dictionary of keytype to key is
        returned.  The keytype will be either C{"ssh-rsa"} or C{"ssh-dss"}.

        @param hostname: the hostname (or IP) to lookup
        @type hostname: str
        @return: keys associated with this host (or C{None})
        @rtype: dict(str, L{PKey})
        """
        class SubDict (UserDict.DictMixin):
            def __init__(self, hostname, entries, hostkeys):
                self._hostname = hostname
                self._entries = entries
                self._hostkeys = hostkeys

            def __getitem__(self, key):
                for e in self._entries:
                    if e.key.get_name() == key:
                        return e.key
                raise KeyError(key)

            def __setitem__(self, key, val):
                for e in self._entries:
                    if e.key is None:
                        continue
                    if e.key.get_name() == key:
                        # replace
                        e.key = val
                        break
                else:
                    # add a new one
                    e = HostKeyEntry([hostname], val)
                    self._entries.append(e)
                    self._hostkeys._entries.append(e)

            def keys(self):
                return [e.key.get_name() for e in self._entries if e.key is not None]

        entries = []
        for e in self._entries:
            for h in e.hostnames:
                if (h.startswith('|1|') and (self.hash_host(hostname, h) == h)) or (h == hostname):
                    entries.append(e)
        if len(entries) == 0:
            return None
        return SubDict(hostname, entries, self)