def create_asset(self, **kwargs):
        """
        Creates an instance of the Asset Service.
        """
        asset = predix.admin.asset.Asset(**kwargs)
        asset.create()

        client_id = self.get_client_id()
        if client_id:
            asset.grant_client(client_id)

        asset.add_to_manifest(self)
        return asset