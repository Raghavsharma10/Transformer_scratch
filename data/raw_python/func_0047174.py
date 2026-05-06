def get_activity_form_for_create(self, objective_id, activity_record_types):
        """Gets the activity form for creating new activities.

        A new form should be requested for each create transaction.

        arg:    objective_id (osid.id.Id): the ``Id`` of the
                ``Objective``
        arg:    activity_record_types (osid.type.Type[]): array of
                activity record types
        return: (osid.learning.ActivityForm) - the activity form
        raise:  NotFound - ``objective_id`` is not found
        raise:  NullArgument - ``objective_id`` or
                ``activity_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ActivityAdminSession.get_activity_form_for_create_template

        if not isinstance(objective_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in activity_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if activity_record_types == []:
            # WHY are we passing objective_bank_id = self._catalog_id below, seems redundant:
            obj_form = objects.ActivityForm(
                objective_bank_id=self._catalog_id,
                objective_id=objective_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.ActivityForm(
                objective_bank_id=self._catalog_id,
                record_types=activity_record_types,
                objective_id=objective_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form