def get_asset(self):
        """
        Returns an instance of the Asset Service.
        """
        import predix.data.asset
        asset = predix.data.asset.Asset()
        return asset