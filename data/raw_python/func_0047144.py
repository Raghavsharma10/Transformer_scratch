def get_objective_form_for_create(self, objective_record_types):
        """Gets the objective form for creating new objectives.

        A new form should be requested for each create transaction.

        arg:    objective_record_types (osid.type.Type[]): array of
                objective record types
        return: (osid.learning.ObjectiveForm) - the objective form
        raise:  NullArgument - ``objective_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.get_resource_form_for_create_template
        for arg in objective_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if objective_record_types == []:
            obj_form = objects.ObjectiveForm(
                objective_bank_id=self._catalog_id,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        else:
            obj_form = objects.ObjectiveForm(
                objective_bank_id=self._catalog_id,
                record_types=objective_record_types,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form