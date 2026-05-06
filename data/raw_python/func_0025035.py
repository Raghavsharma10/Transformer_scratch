def grant_client(self, client_id, read=True, write=True):
        """
        Grant the given client id all the scopes and authorities
        needed to work with the timeseries service.
        """
        scopes = ['openid']
        authorities = ['uaa.resource']

        if write:
            for zone in self.service.settings.data['ingest']['zone-token-scopes']:
                scopes.append(zone)
                authorities.append(zone)

        if read:
            for zone in self.service.settings.data['query']['zone-token-scopes']:
                scopes.append(zone)
                authorities.append(zone)

        self.service.uaa.uaac.update_client_grants(client_id, scope=scopes,
                authorities=authorities)

        return self.service.uaa.uaac.get_client(client_id)