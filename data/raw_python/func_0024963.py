def create_acs(self, **kwargs):
        """
        Creates an instance of the Asset Service.
        """
        acs = predix.admin.acs.AccessControl(**kwargs)
        acs.create()

        client_id = self.get_client_id()
        if client_id:
            acs.grant_client(client_id)

        acs.grant_client(client_id)
        acs.add_to_manifest(self)
        return acs