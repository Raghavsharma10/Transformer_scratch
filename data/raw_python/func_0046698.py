def get_grade_system_form_for_create(self, grade_system_record_types):
        """Gets the grade system form for creating new grade systems.

        A new form should be requested for each create transaction.

        arg:    grade_system_record_types (osid.type.Type[]): array of
                grade system types
        return: (osid.grading.GradeSystemForm) - the grade system form
        raise:  NullArgument - ``grade_system_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.get_resource_form_for_create_template
        for arg in grade_system_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if grade_system_record_types == []:
            obj_form = objects.GradeSystemForm(
                gradebook_id=self._catalog_id,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        else:
            obj_form = objects.GradeSystemForm(
                gradebook_id=self._catalog_id,
                record_types=grade_system_record_types,
                runtime=self._runtime,
                effective_agent_id=self.get_effective_agent_id(),
                proxy=self._proxy)
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form