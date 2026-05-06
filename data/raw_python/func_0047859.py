def delete_asset_content(self, asset_content_id=None):
        """Deletes content from an ``Asset``.

        :param asset_content_id: the ``Id`` of the ``AssetContent``
        :type asset_content_id: ``osid.id.Id``
        :raise: ``NotFound`` -- ``asset_content_id`` is not found
        :raise: ``NullArgument`` -- ``asset_content_id`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_content_id is None:
            raise NullArgument()
        asset = None
        for a in AssetLookupSession(self._repository_id,
                                    proxy=self._proxy,
                                    runtime=self._runtime).get_assets():
            i = 0
            # might want to set plenary view
            # to assure ordering?
            for ac in a.get_asset_contents():
                if ac.get_id() == asset_content_id:
                    asset = a
                    asset_content = ac
                    index = i
                i += 1
        if asset is None:
            raise NotFound()

        asset._my_map['assetContents'].pop(index)
        url_path = construct_url('assets',
                                 bank_id=self._catalog_idstr)
        try:
            result = self._put_request(url_path, asset._my_map)
        except Exception:
            raise