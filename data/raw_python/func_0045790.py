def update_log(self, log_form):
        """Updates an existing log.

        arg:    log_form (osid.logging.LogForm): the form containing the
                elements to be updated
        raise:  IllegalState - ``log_form`` already used in an update
                transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``log_id`` or ``log_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``log_form`` did not originate from
                ``get_log_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinAdminSession.update_bin_template
        if self._catalog_session is not None:
            return self._catalog_session.update_catalog(catalog_form=log_form)
        collection = JSONClientValidated('logging',
                                         collection='Log',
                                         runtime=self._runtime)
        if not isinstance(log_form, ABCLogForm):
            raise errors.InvalidArgument('argument type is not an LogForm')
        if not log_form.is_for_update():
            raise errors.InvalidArgument('the LogForm is for update only, not create')
        try:
            if self._forms[log_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('log_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('log_form did not originate from this session')
        if not log_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(log_form._my_map)  # save is deprecated - change to replace_one

        self._forms[log_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned
        return objects.Log(osid_object_map=log_form._my_map, runtime=self._runtime, proxy=self._proxy)