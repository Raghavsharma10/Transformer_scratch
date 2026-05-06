def timeline(self, uri):
        '''Get the domain tagging timeline for a given uri. 
        Could be a domain, ip, or url.
        For details, see https://docs.umbrella.com/investigate-api/docs/timeline
        '''
        uri = self._uris["timeline"].format(uri)
        resp_json = self.get_parse(uri)

        return resp_json