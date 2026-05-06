def create_authorization(self, authorization_form):
        """Creates a new explicit ``Authorization``.

        arg:    authorization_form
                (osid.authorization.AuthorizationForm): the
                authorization form
        return: (osid.authorization.Authorization) - ``t`` he new
                ``Authorization``
        raise:  IllegalState - ``authorization_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``authorization_form`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``authorization_form`` did not originate
                from this service
        *compliance: mandatory -- This method must be implemented.*

        """
        # TODO: not using the create_resource template
        # because want to prevent duplicate authorizations
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        if not isinstance(authorization_form, ABCAuthorizationForm):
            raise errors.InvalidArgument('argument type is not an AuthorizationForm')
        if authorization_form.is_for_update():
            raise errors.InvalidArgument('the AuthorizationForm is for update only, not create')
        try:
            if self._forms[authorization_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('authorization_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('authorization_form did not originate from this session')
        if not authorization_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')

        # try to check first here
        try:
            osid_map = collection.find_one({"agentId": authorization_form._my_map['agentId'],
                                            "functionId": authorization_form._my_map['functionId'],
                                            "qualifierId": authorization_form._my_map['qualifierId'],
                                            "assignedVaultIds": authorization_form._my_map['assignedVaultIds']})
            osid_map['startDate'] = authorization_form._my_map['startDate']
            osid_map['endDate'] = authorization_form._my_map['endDate']
            collection.save(osid_map)
        except errors.NotFound:
            insert_result = collection.insert_one(authorization_form._my_map)

            self._forms[authorization_form.get_id().get_identifier()] = CREATED
            osid_map = collection.find_one({'_id': insert_result.inserted_id})
        result = objects.Authorization(
            osid_object_map=osid_map,
            runtime=self._runtime,
            proxy=self._proxy)

        return result