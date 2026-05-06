def get_assets_by_ids(self, asset_ids=None):
        """Gets an ``AssetList`` corresponding to the given ``IdList``.

        In plenary mode, the returned list contains all of the assets
        specified in the ``Id`` list, in the order of the list,
        including duplicates, or an error results if an ``Id`` in the
        supplied list is not found or inaccessible. Otherwise,
        inaccessible ``Assets`` may be omitted from the list and may
        present the elements in any order including returning a unique
        set.

        :param asset_ids: the list of ``Ids`` to retrieve
        :type asset_ids: ``osid.id.IdList``
        :return: the returned ``Asset list``
        :rtype: ``osid.repository.AssetList``
        :raise: ``NotFound`` -- an ``Id`` was not found
        :raise: ``NullArgument`` -- ``asset_ids`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_ids is None:
            raise NullArgument()
        assets = []
        for i in asset_ids:
            asset = None
            url_path = construct_url('assets',
                                     bank_id=self._catalog_idstr,
                                     asset_id=i)
            try:
                asset = self._get_request(url_path)
            except (NotFound, OperationFailed):
                if self._objective_view == PLENARY:
                    raise
                else:
                    pass
            if asset:
                if not (self._asset_view == COMPARATIVE and
                        asset in assets):
                    assets.append(asset)
        return objects.AssetList(assets)