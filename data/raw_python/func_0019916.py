def as_for_ip(self, ip):
        '''Gets the AS information for a given IP address.'''
        if not Investigate.IP_PATTERN.match(ip):
            raise Investigate.IP_ERR

        uri = self._uris["as_for_ip"].format(ip)
        resp_json = self.get_parse(uri)

        return resp_json