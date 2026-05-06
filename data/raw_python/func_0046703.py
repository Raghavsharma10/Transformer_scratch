def get_grade_form_for_create(self, grade_system_id, grade_record_types):
        """Gets the grade form for creating new grades.

        A new form should be requested for each create transaction.

        arg:    grade_system_id (osid.id.Id): the ``Id`` of a
                ``GradeSystem``
        arg:    grade_record_types (osid.type.Type[]): array of grade
                recod types
        return: (osid.grading.GradeForm) - the grade form
        raise:  NotFound - ``grade_system_id`` is not found
        raise:  NullArgument - ``grade_system_id`` or
                ``grade_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityAdminSession.get_activity_form_for_create_template

        if not isinstance(grade_system_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in grade_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if grade_record_types == []:
            # WHY are we passing gradebook_id = self._catalog_id below, seems redundant:
            obj_form = objects.GradeForm(
                gradebook_id=self._catalog_id,
                grade_system_id=grade_system_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.GradeForm(
                gradebook_id=self._catalog_id,
                record_types=grade_record_types,
                grade_system_id=grade_system_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form