def get_asset_id(self):
        """Gets the ``Asset Id`` corresponding to this content.

        return: (osid.id.Id) - the asset ``Id``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.Activity.get_objective_id
        if not bool(self._my_map['assetId']):
            raise errors.IllegalState('asset empty')
        return Id(self._my_map['assetId'])