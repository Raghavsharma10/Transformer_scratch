def target(self):
        """
        Target the current space for any forthcoming Cloud Foundry
        operations.
        """
        # MAINT: I don't like this, but will deal later
        os.environ['PREDIX_SPACE_GUID'] = self.guid
        os.environ['PREDIX_SPACE_NAME'] = self.name
        os.environ['PREDIX_ORGANIZATION_GUID'] = self.org.guid
        os.environ['PREDIX_ORGANIZATION_NAME'] = self.org.name