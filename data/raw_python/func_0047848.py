def update_asset(self, asset_form=None):
        """Updates an existing asset.

        :param asset_form: the form containing the elements to be updated
        :type asset_form: ``osid.repository.AssetForm``
        :raise: ``IllegalState`` -- ``asset_form`` already used in anupdate transaction
        :raise: ``InvalidArgument`` -- the form contains an invalid value
        :raise: ``NullArgument`` -- ``asset_form`` is ``null``
        :raise: ``OperationFailed`` -- unable to complete request
        :raise: ``PermissionDenied`` -- authorization failure
        :raise: ``Unsupported`` -- ``asset_form`` did not originate from ``get_asset_form_for_update()``

        *compliance: mandatory -- This method must be implemented.*

        """
        if asset_form is None:
            raise NullArgument()
        if not isinstance(asset_form, abc_repository_objects.AssetForm):
            raise InvalidArgument('argument type is not an AssetForm')
        if not asset_form.is_for_update():
            raise InvalidArgument('form is for create only, not update')
        try:
            if self._forms[asset_form.get_id().get_identifier()] == UPDATED:
                raise IllegalState('form already used in an update transaction')
        except KeyError:
            raise Unsupported('form did not originate from this session')
        if not asset_form.is_valid():
            raise InvalidArgument('one or more of the form elements is invalid')

        url_path = construct_url('assets',
                                 bank_id=self._catalog_idstr)
        try:
            result = self._put_request(url_path, asset_form._my_map)
        except Exception:
            raise  # OperationFailed()
        self._forms[asset_form.get_id().get_identifier()] = UPDATED
        return objects.Asset(result)