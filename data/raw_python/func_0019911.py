def domain_whois_history(self, domain, limit=None):
        '''Gets whois history for a domain'''

        params = dict()
        if limit is not None:
            params['limit'] = limit

        uri = self._uris["whois_domain_history"].format(domain)
        resp_json = self.get_parse(uri, params)
        return resp_json