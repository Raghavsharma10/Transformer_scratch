def related(self, domain):
        '''Get the related domains of the given domain.

        For details, see https://investigate.umbrella.com/docs/api#relatedDomains
        '''
        uri = self._uris["related"].format(domain)
        return self.get_parse(uri)