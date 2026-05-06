def _get_spaces(self):
        """
        Get the marketplace services.
        """
        guid = self.api.config.get_organization_guid()
        uri = '/v2/organizations/%s/spaces' % (guid)
        return self.api.get(uri)