def get_asset_content_ids(self):
        """Gets the content ``Ids`` of this asset.

        return: (osid.id.IdList) - the asset content ``Ids``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.Asset.get_asset_content_ids_template
        id_list = []
        for asset_content in self.get_asset_contents():
            id_list.append(asset_content.get_id())
        return IdList(id_list)