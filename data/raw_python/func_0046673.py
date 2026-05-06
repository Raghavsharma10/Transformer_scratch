def update_vault(self, vault_form):
        """Updates an existing vault.

        arg:    vault_form (osid.authorization.VaultForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``vault_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``vault_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``vault_form`` did not originate from
                ``get_vault_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=vault_form)
        collection = JSONClientValidated('authorization',
                                         collection='Vault',
                                         runtime=self._runtime)
        if not isinstance(vault_form, ABCVaultForm):
            raise errors.InvalidArgument('argument type is not an VaultForm')
        if not vault_form.is_for_update():
            raise errors.InvalidArgument('the VaultForm is for update only, not create')
        try:
            if self._forms[vault_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('vault_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('vault_form did not originate from this session')
        if not vault_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(vault_form._my_map)  # save is deprecated - change to replace_one

        self._forms[vault_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Vault(osid_object_map=vault_form._my_map, runtime=self._runtime, proxy=self._proxy)