def get_branding(self):
        """Gets a branding, such as an image or logo, expressed using the ``Asset`` interface.

        return: (osid.repository.AssetList) - a list of assets
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        mgr = self.my_osid_object._get_provider_manager('REPOSITORY')
        lookup_session = mgr.get_asset_lookup_session()
        lookup_session.use_federated_repository_view()
        return lookup_session.get_assets_by_ids(self.get_branding_ids())