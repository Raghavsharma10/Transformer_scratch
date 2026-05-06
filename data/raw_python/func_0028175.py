def get_organizations(self, permission='read'):
        # type: (str) -> List['Organization']
        """Get organizations in HDX that this user is a member of.

        Args:
            permission (str): Permission to check for. Defaults to 'read'.

        Returns:
            List[Organization]: List of organizations in HDX that this user is a member of
        """
        success, result = self._read_from_hdx('user', self.data['name'], 'id', self.actions()['listorgs'],
                                              permission=permission)
        organizations = list()
        if success:
            for organizationdict in result:
                organization = hdx.data.organization.Organization.read_from_hdx(organizationdict['id'])
                organizations.append(organization)
        return organizations