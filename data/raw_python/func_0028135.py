def set_organization(self, organization):
        # type: (Union[hdx.data.organization.Organization,Dict,str]) -> None
        """Set the dataset's organization.

         Args:
             organization (Union[Organization,Dict,str]): Either an Organization id or Organization metadata from an Organization object or dictionary.
         Returns:
             None
        """
        if isinstance(organization, hdx.data.organization.Organization) or isinstance(organization, dict):
            if 'id' not in organization:
                organization = hdx.data.organization.Organization.read_from_hdx(organization['name'], configuration=self.configuration)
            organization = organization['id']
        elif not isinstance(organization, str):
            raise HDXError('Type %s cannot be added as a organization!' % type(organization).__name__)
        if is_valid_uuid(organization) is False and organization != 'hdx':
            raise HDXError('%s is not a valid organization id!' % organization)
        self.data['owner_org'] = organization