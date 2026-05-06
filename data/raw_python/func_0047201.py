def create_proficiency(self, proficiency_form):
        """Creates a new ``Proficiency``.

        A new form should be requested for each create transaction.

        arg:    proficiency_form (osid.learning.ProficiencyForm): the
                form for this ``Proficiency``
        return: (osid.learning.Proficiency) - the new ``Proficiency``
        raise:  IllegalState - ``proficiency_form`` already used in a
                create transaction
        raise:  InvalidArgument - one or more of the form elements is
                invalid
        raise:  NullArgument - ``proficiency_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``proficiency_form`` did not originate
                from ``get_proficiency_form_for_create()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.create_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        if not isinstance(proficiency_form, ABCProficiencyForm):
            raise errors.InvalidArgument('argument type is not an ProficiencyForm')
        if proficiency_form.is_for_update():
            raise errors.InvalidArgument('the ProficiencyForm is for update only, not create')
        try:
            if self._forms[proficiency_form.get_id().get_identifier()] == CREATED:
                raise errors.IllegalState('proficiency_form already used in a create transaction')
        except KeyError:
            raise errors.Unsupported('proficiency_form did not originate from this session')
        if not proficiency_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        insert_result = collection.insert_one(proficiency_form._my_map)

        self._forms[proficiency_form.get_id().get_identifier()] = CREATED
        result = objects.Proficiency(
            osid_object_map=collection.find_one({'_id': insert_result.inserted_id}),
            runtime=self._runtime,
            proxy=self._proxy)

        return result