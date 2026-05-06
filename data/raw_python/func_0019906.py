def cooccurrences(self, domain):
        '''Get the cooccurrences of the given domain.

        For details, see https://investigate.umbrella.com/docs/api#co-occurrences
        '''
        uri = self._uris["cooccurrences"].format(domain)
        return self.get_parse(uri)