def update_proficiency(self, proficiency_form):
        """Updates an existing proficiency.

        arg:    proficiency_form (osid.learning.ProficiencyForm): the
                form containing the elements to be updated
        raise:  IllegalState - ``proficiency_form`` already used in an
                update transaction
        raise:  InvalidArgument - the form contains an invalid value
        raise:  NullArgument - ``proficiency_form`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - ``proficiency_form`` did not originate
                from ``get_proficiency_form_for_update()``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceAdminSession.update_resource_template
        collection = JSONClientValidated('learning',
                                         collection='Proficiency',
                                         runtime=self._runtime)
        if not isinstance(proficiency_form, ABCProficiencyForm):
            raise errors.InvalidArgument('argument type is not an ProficiencyForm')
        if not proficiency_form.is_for_update():
            raise errors.InvalidArgument('the ProficiencyForm is for update only, not create')
        try:
            if self._forms[proficiency_form.get_id().get_identifier()] == UPDATED:
                raise errors.IllegalState('proficiency_form already used in an update transaction')
        except KeyError:
            raise errors.Unsupported('proficiency_form did not originate from this session')
        if not proficiency_form.is_valid():
            raise errors.InvalidArgument('one or more of the form elements is invalid')
        collection.save(proficiency_form._my_map)

        self._forms[proficiency_form.get_id().get_identifier()] = UPDATED

        # Note: this is out of spec. The OSIDs don't require an object to be returned:
        return objects.Proficiency(
            osid_object_map=proficiency_form._my_map,
            runtime=self._runtime,
            proxy=self._proxy)