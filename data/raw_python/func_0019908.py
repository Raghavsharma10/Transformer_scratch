def security(self, domain):
        '''Get the Security Information for the given domain.

        For details, see https://investigate.umbrella.com/docs/api#securityInfo
        '''
        uri = self._uris["security"].format(domain)
        return self.get_parse(uri)