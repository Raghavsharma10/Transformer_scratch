def get_vault_form_for_update(self, vault_id):
        """Gets the vault form for updating an existing vault.

        A new vault form should be requested for each update
        transaction.

        arg:    vault_id (osid.id.Id): the ``Id`` of the ``Vault``
        return: (osid.authorization.VaultForm) - the vault form
        raise:  NotFound - ``vault_id`` is not found
        raise:  NullArgument - ``vault_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.get_bin_form_for_update_template
        if self._catalog_session is not None:
            return self._catalog_session.get_catalog_form_for_update(catalog_id=vault_id)
        collection = JSONClientValidated('authorization',
                                         collection='Vault',
                                         runtime=self._runtime)
        if not isinstance(vault_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        result = collection.find_one({'_id': ObjectId(vault_id.get_identifier())})

        cat_form = objects.VaultForm(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)
        self._forms[cat_form.get_id().get_identifier()] = not UPDATED

        return cat_form