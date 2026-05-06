def create_vault(self, vault_form):
        """Creates a new ``Vault``.

        arg:    vault_form (osid.authorization.VaultForm): the form for
                this ``Vault``
        return: (osid.authorization.Vault) - the new ``Vault``
        raise:  IllegalState - ``vault_form`` already used in a create
                transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``vault_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``vault_form`` did not originate from
                ``get_vault_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.create_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.create_catalog(catalog_form=vault_form)
        collection = JSONClientValidated('authorization',
                                         collection='Vault',
                                         runtime=self._runtime)
        if not isinstance(vault_form, ABCVaultForm):
            raise errors.InvalidArgument('argument type is not an VaultForm')
        if vault_form.is_for_update():
            raise errors.InvalidArgument('the VaultForm is for update only, not create')
        try:
            if self._forms[vault_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('vault_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('vault_form did not originate from this session')
        if not vault_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(vault_form._my_map)

        self._forms[vault_form.get_id().get_identifier()] = CREATED
        result = objects.Vault(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result