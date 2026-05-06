def update_objective_bank(self, objective_bank_form):
        """Updates an existing objective bank.

        arg:    objective_bank_form (osid.learning.ObjectiveBankForm):
                the form containing the elements to be updated
        raise:  IllegalState - ``objective_bank_form`` already used in
                an update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``objective_bank_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``objective_bank_form did not originate
                from get_objective_bank_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=objective_bank_form)
        collection = JSONClientValidated('learning',
                                         collection='ObjectiveBank',
                                         runtime=self._runtime)
        if not isinstance(objective_bank_form, ABCObjectiveBankForm):
            raise errors.InvalidArgument('argument type is not an ObjectiveBankForm')
        if not objective_bank_form.is_for_update():
            raise errors.InvalidArgument('the ObjectiveBankForm is for update only, not create')
        try:
            if self._forms[objective_bank_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('objective_bank_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('objective_bank_form did not originate from this session')
        if not objective_bank_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(objective_bank_form._my_map)  # save is deprecated - change to replace_one

        self._forms[objective_bank_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.ObjectiveBank(osid_object_map=objective_bank_form._my_map, runtime=self._runtime, proxy=self._proxy)