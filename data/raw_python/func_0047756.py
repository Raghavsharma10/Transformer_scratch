def get_resource_form_for_create(self, resource_record_types):
        """Gets the resource form for creating new resources.

        A new form should be requested for each create transaction.

        arg:    resource_record_types (osid.type.Type[]): array of
                resource record types
        return: (osid.resource.ResourceForm) - the resource form
        raise:  NullArgument - ``resource_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form with requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.get_resource_form_for_create_template
        for arg in resource_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if resource_record_types == []:
            obj_form = objects.ResourceForm(
                bin_id=self._catalog_id,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        else:
            obj_form = objects.ResourceForm(
                bin_id=self._catalog_id,
                record_types=resource_record_types,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form