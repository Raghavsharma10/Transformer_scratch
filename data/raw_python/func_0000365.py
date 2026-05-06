def get_organization(self, id=None, name=None):
        """
        Gets existing and accessible organization
        :rtype: Organization
        """
        log.info("Picking organization: %s (%s)" % (name, id))
        return self.organizations[id or name]