def get_authorization_form_for_create_for_agent(self, agent_id, function_id, qualifier_id, authorization_record_types):
        """Gets the authorization form for creating new authorizations.

        A new form should be requested for each create transaction.

        arg:    agent_id (osid.id.Id): the agent ``Id``
        arg:    function_id (osid.id.Id): the function ``Id``
        arg:    qualifier_id (osid.id.Id): the qualifier ``Id``
        arg:    authorization_record_types (osid.type.Type[]): array of
                authorization record types
        return: (osid.authorization.AuthorizationForm) - the
                authorization form
        raise:  NotFound - ``agent_id, function_id`` or ``qualifier_id``
                is not found
        raise:  NullArgument - ``agent_id, function_id, qualifier_id``
                or ``authorization_record_types`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form with requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(agent_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        if not isinstance(function_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        if not isinstance(qualifier_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in authorization_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if authorization_record_types == []:
            # WHY are we passing vault_id = self._catalog_id below, seems redundant:
            # We probably also don't need to send agent_id. The form can now get that from the proxy
            obj_form = objects.AuthorizationForm(
                vault_id=self._catalog_id,
                agent_id=agent_id,
                function_id=function_id,
                qualifier_id=qualifier_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.AuthorizationForm(
                vault_id=self._catalog_id,
                record_types=authorization_record_types,
                agent_id=agent_id,
                function_id=function_id,
                qualifier_id=qualifier_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form