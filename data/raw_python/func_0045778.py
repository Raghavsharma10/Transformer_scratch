def update_log_entry(self, log_entry_form):
        """Updates an existing log entry.

        arg:    log_entry_form (osid.logging.LogEntryForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``log_entry_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``log_entry_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``log_entry_form`` did not originate from
                ``get_log_entry_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('logging',
                                         collection='LogEntry',
                                         runtime=self._runtime)
        if not isinstance(log_entry_form, ABCLogEntryForm):
            raise errors.InvalidArgument('argument type is not an LogEntryForm')
        if not log_entry_form.is_for_update():
            raise errors.InvalidArgument('the LogEntryForm is for update only, not create')
        try:
            if self._forms[log_entry_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('log_entry_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('log_entry_form did not originate from this session')
        if not log_entry_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(log_entry_form._my_map)

        self._forms[log_entry_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.LogEntry(
            osid_object_map=log_entry_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)