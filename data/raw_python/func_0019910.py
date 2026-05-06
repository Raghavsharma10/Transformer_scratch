def domain_whois(self, domain):
        '''Gets whois information for a domain'''
        uri = self._uris["whois_domain"].format(domain)
        resp_json = self.get_parse(uri)
        return resp_json