def update_asset_content(self, asset_content_form=None):
        """Updates an existing asset.

        :param asset_content_form: the form containing the elements to be updated
        :type asset_content_form: ``osid.repository.AssetContentForm``
        :raise: ``IllegalState`` -- ``asset_content_form`` already used in an update transaction
        :raise: ``InvalidArgument`` -- the form contains an invalid value
        :raise: ``NullArgument`` -- ``asset_form`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- ``asset_content_form`` did not originate from ``get_asset_content_form_for_update()``

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_content_form is None:
            raise NullArgument()
        if not isinstance(asset_content_form, abc_repository_objects.AssetContentForm):
            raise InvalidArgument('argument type is not an AssetContentForm')
        if not asset_content_form.is_for_update():
            raise InvalidArgument('form is for create only, not update')
        try:
            if self._forms[asset_content_form.get_id().get_identifier()] == CREATED:
                raise IllegalState('form already used in a create transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not asset_content_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = construct_url('assets',
                                 bank_id=self._catalog_idstr,
                                 asset_id=asset_content_form._asset_id)
        asset = objects.Asset(self._get_request(url_path))
        index = 0
        for ac in asset.get_asset_contents():
            if str(ac.get_id()) == asset_content_form._my_map['id']:
                break
            index += 1
        asset._my_map['assetContents'].pop(index)
        asset._my_map['assetContents'].insert(index, asset_content_form._my_map)
        url_path = construct_url('assets',
                                 bank_id=self._catalog_idstr)
        try:
            result = self._put_request(url_path, asset._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[asset_content_form.get_id().get_identifier()] = CREATED
        return objects.AssetContent(asset_content_form._my_map)