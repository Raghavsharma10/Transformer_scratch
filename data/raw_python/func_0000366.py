def get_or_create_organization(self, id=None, name=None):
        """
        Gets existing or creates new organization
        :rtype: Organization
        """
        if id:
            return self.get_organization(id)
        else:
            assert name
            try:
                return self.get_organization(name=name)
            except exceptions.NotFoundError:
                return self.create_organization(name)