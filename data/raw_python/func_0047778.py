def update_bin(self, bin_form):
        """Updates an existing bin.

        arg:    bin_form (osid.resource.BinForm): the form containing
                the elements to be updated
        raise:  IllegalState - ``bin_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``bin_id`` or ``bin_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``bin_form`` did not originate from
                ``get_bin_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=bin_form)
        collection = JSONClientValidated('resource',
                                         collection='Bin',
                                         runtime=self._runtime)
        if not isinstance(bin_form, ABCBinForm):
            raise errors.InvalidArgument('argument type is not an BinForm')
        if not bin_form.is_for_update():
            raise errors.InvalidArgument('the BinForm is for update only, not create')
        try:
            if self._forms[bin_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('bin_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('bin_form did not originate from this session')
        if not bin_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(bin_form._my_map)  # save is deprecated - change to replace_one

        self._forms[bin_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Bin(osid_object_map=bin_form._my_map, runtime=self._runtime, proxy=self._proxy)