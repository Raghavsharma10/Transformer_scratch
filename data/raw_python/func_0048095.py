def update_bank(self, bank_form):
        """Updates an existing bank.

        arg:    bank_form (osid.assessment.BankForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``bank_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``bank_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        raise:  Unsupported - ``bank_form`` did not originate from
                ``get_bank_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=bank_form)
        collection = JSONClientValidated('assessment',
                                         collection='Bank',
                                         runtime=self._runtime)
        if not isinstance(bank_form, ABCBankForm):
            raise errors.InvalidArgument('argument type is not an BankForm')
        if not bank_form.is_for_update():
            raise errors.InvalidArgument('the BankForm is for update only, not create')
        try:
            if self._forms[bank_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('bank_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('bank_form did not originate from this session')
        if not bank_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(bank_form._my_map)  # save is deprecated - change to replace_one

        self._forms[bank_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Bank(osid_object_map=bank_form._my_map, runtime=self._runtime, proxy=self._proxy)