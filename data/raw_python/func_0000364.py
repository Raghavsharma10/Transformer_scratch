def create_organization(self, name):
        """
        Creates new organization
        :rtype: Organization
        """
        org = Organization.new(name, self._router)
        assert org.ready(), "Organization {} hasn't got ready after creation".format(name)
        return org