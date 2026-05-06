def _get_hosts_from_names(self, names):
        """ validate hostnames from a list of names
        """
        result = set()
        hosts = map(lambda x: x.strip(), names.split(','))
        for h in hosts:
            if valid_hostname(h.split(':')[0]):
                result.add(h if ':' in h else '%s:%d' % (h, self.PORT))
            else:
                raise conferr('Invalid hostname: %s' % h.split(':')[0])
        return list(result)