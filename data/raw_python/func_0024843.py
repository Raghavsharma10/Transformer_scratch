def grant_client(self, client_id):
        """
        Grant the given client id all the scopes and authorities
        needed to work with the access control service.
        """
        zone = self.service.settings.data['zone']['oauth-scope']

        scopes = ['openid', zone,
                  'acs.policies.read', 'acs.attributes.read',
                  'acs.policies.write', 'acs.attributes.write']

        authorities = ['uaa.resource', zone,
                  'acs.policies.read', 'acs.policies.write',
                  'acs.attributes.read', 'acs.attributes.write']

        self.service.uaa.uaac.update_client_grants(client_id, scope=scopes,
                authorities=authorities)

        return self.service.uaa.uaac.get_client(client_id)