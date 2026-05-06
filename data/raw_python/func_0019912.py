def ns_whois(self, nameservers, limit=DEFAULT_LIMIT, offset=DEFAULT_OFFSET, sort_field=DEFAULT_SORT):
        '''Gets the domains that have been registered with a nameserver or
        nameservers'''
        if not isinstance(nameservers, list):
            uri = self._uris["whois_ns"].format(nameservers)
            params = {'limit': limit, 'offset': offset, 'sortField': sort_field}
        else:
            uri = self._uris["whois_ns"].format('')
            params = {'emailList' : ','.join(nameservers), 'limit': limit, 'offset': offset, 'sortField': sort_field}

        resp_json = self.get_parse(uri, params=params)
        return resp_json