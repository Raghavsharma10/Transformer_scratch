def lookup(self, hostname):
        """
        Return a dict of config options for a given hostname.

        The host-matching rules of OpenSSH's C{ssh_config} man page are used,
        which means that all configuration options from matching host
        specifications are merged, with more specific hostmasks taking
        precedence. In other words, if C{"Port"} is set under C{"Host *"}
        and also C{"Host *.example.com"}, and the lookup is for
        C{"ssh.example.com"}, then the port entry for C{"Host *.example.com"}
        will win out.

        The keys in the returned dict are all normalized to lowercase (look for
        C{"port"}, not C{"Port"}. No other processing is done to the keys or
        values.

        @param hostname: the hostname to lookup
        @type hostname: str
        """
        matches = [x for x in self._config if fnmatch.fnmatch(hostname, x['host'])]
        # Move * to the end
        _star = matches.pop(0)
        matches.append(_star)
        ret = {}
        for m in matches:
            for k,v in m.iteritems():
                if not k in ret:
                    ret[k] = v
        ret = self._expand_variables(ret, hostname)
        del ret['host']
        return ret