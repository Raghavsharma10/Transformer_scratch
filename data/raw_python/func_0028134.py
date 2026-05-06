def get_organization(self):
        # type: () -> hdx.data.organization.Organization
        """Get the dataset's organization.

         Returns:
             Organization: Dataset's organization
        """
        return hdx.data.organization.Organization.read_from_hdx(self.data['owner_org'], configuration=self.configuration)