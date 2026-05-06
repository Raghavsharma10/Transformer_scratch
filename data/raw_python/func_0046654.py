def update_authorization(self, authorization_form):
        """Updates an existing authorization.

        arg:    authorization_form
                (osid.authorization.AuthorizationForm): the
                authorization ``Id``
        raise:  IllegalState - ``authorization_form`` already used in an
                update transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``authorization_form`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``authorization_form`` did not originate
                from ``get_authorization_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        if not isinstance(authorization_form, ABCAuthorizationForm):
            raise errors.InvalidArgument('argument type is not an AuthorizationForm')
        if not authorization_form.is_for_update():
            raise errors.InvalidArgument('the AuthorizationForm is for update only, not create')
        try:
            if self._forms[authorization_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('authorization_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('authorization_form did not originate from this session')
        if not authorization_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(authorization_form._my_map)

        self._forms[authorization_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Authorization(
            osid_object_map=authorization_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)