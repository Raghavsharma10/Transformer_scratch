def update_resource(self, resource_form):
        """Updates an existing resource.

        arg:    resource_form (osid.resource.ResourceForm): the form
                containing the elements to be updated
        raise:  IllegalState - ``resource_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``resource_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``resource_form`` did not originate from
                ``get_resource_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        if not isinstance(resource_form, ABCResourceForm):
            raise errors.InvalidArgument('argument type is not an ResourceForm')
        if not resource_form.is_for_update():
            raise errors.InvalidArgument('the ResourceForm is for update only, not create')
        try:
            if self._forms[resource_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('resource_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('resource_form did not originate from this session')
        if not resource_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(resource_form._my_map)

        self._forms[resource_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Resource(
            osid_object_map=resource_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)