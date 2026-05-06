def get_asset_content_form_for_update(self, asset_content_id=None):
        """Gets the asset form for updating content for an existing asset.

        A new asset content form should be requested for each update
        transaction.

        :param asset_content_id: the ``Id`` of the ``AssetContent``
        :type asset_content_id: ``osid.id.Id``
        :return: the asset content form
        :rtype: ``osid.repository.AssetContentForm``
        :raise: ``NotFound`` -- ``asset_content_id`` is not found
        :raise: ``NullArgument`` -- ``asset_content_id`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_content_id is None:
            raise NullArgument()
        asset = None
        for a in AssetLookupSession(self._repository_id,
                                    proxy=self._proxy,
                                    runtime=self._runtime).get_assets():
            # might want to set plenary view
            # to assure ordering?
            for ac in a.get_asset_contents():
                if ac.get_id() == asset_content_id:
                    asset = a
                    asset_content = ac
        if asset is None:
            raise NotFound()
        asset_content_form = objects.AssetContentForm(asset_content._my_map, asset_id=asset.get_id())
        self._forms[asset_content_form.get_id().get_identifier()] = not UPDATED
        return asset_content_form