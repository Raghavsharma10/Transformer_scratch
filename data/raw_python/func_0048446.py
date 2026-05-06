def update_catalog(self, catalog_form):
        """Updates an existing catalog.

        arg:    catalog_form (osid.cataloging.CatalogForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``catalog_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``catalog_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``catalog_form`` did not originate from
                ``get_catalog_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=catalog_form)
        collection = JSONClientValidated('cataloging',
                                         collection='Catalog',
                                         runtime=self._runtime)
        if not isinstance(catalog_form, ABCCatalogForm):
            raise errors.InvalidArgument('argument type is not an CatalogForm')
        if not catalog_form.is_for_update():
            raise errors.InvalidArgument('the CatalogForm is for update only, not create')
        try:
            if self._forms[catalog_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('catalog_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('catalog_form did not originate from this session')
        if not catalog_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(catalog_form._my_map)  # save is deprecated - change to replace_one

        self._forms[catalog_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Catalog(osid_object_map=catalog_form._my_map, runtime=self._runtime, proxy=self._proxy)