def create_asset_content(self, asset_content_form=None):
        """Creates new ``AssetContent`` for a given asset.

        :param asset_content_form: the form for this ``AssetContent``
        :type asset_content_form: ``osid.repository.AssetContentForm``
        :return: the new ``AssetContent``
        :rtype: ``osid.repository.AssetContent``
        :raise: ``IllegalState`` -- ``asset_content_form`` already used in a create transaction
        :raise: ``InvalidArgument`` -- one or more of the form elements is invalid
        :raise: ``NullArgument`` -- ``asset_content_form`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- ``asset_content_form`` did not originate from ``get_asset_content_form_for_create()``

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_content_form is None:
            raise NullArgument()
        if not isinstance(asset_content_form, abc_repository_objects.AssetContentForm):
            raise InvalidArgument('argument type is not an AssetContentForm')
        if asset_content_form.is_for_update():
            raise InvalidArgument('form is for update only, not create')
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
        previous_contents = asset._my_map['assetContents']
        previous_content_ids = [c['id'] for c in previous_contents]
        asset._my_map['assetContents'].append(asset_content_form._my_map)
        url_path = construct_url('assets',
                                 bank_id=self._catalog_idstr)
        try:
            result = self._put_request(url_path, asset._my_map)
        except Exception:
            raise  # OperationFailed
        self._forms[asset_content_form.get_id().get_identifier()] = CREATED
        content = result['assetContents']
        if len(content) == 1:
            return objects.AssetContent(content[0])
        else:
            # Assumes that in the split second this requires,
            # no one else creates a new asset content for this
            # asset...
            for c in content:
                if c['id'] not in previous_content_ids:
                    return objects.AssetContent(c)