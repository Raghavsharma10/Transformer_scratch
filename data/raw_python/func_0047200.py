def get_proficiency_form_for_create(self, objective_id, resource_id, proficiency_record_types):
        """Gets the proficiency form for creating new proficiencies.

        A new form should be requested for each create transaction.

        arg:    objective_id (osid.id.Id): the ``Id`` of the
                ``Objective``
        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
        arg:    proficiency_record_types (osid.type.Type[]): array of
                proficiency record types
        return: (osid.learning.ProficiencyForm) - the proficiency form
        raise:  NotFound - ``objective_id`` or ``resource_id`` is not
                found
        raise:  NullArgument - ``objective_id, resource_id,`` or
                ``proficieny_record_types`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - unable to get form for requested record
                types
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipAdminSession.get_relationship_form_for_create
        # These really need to be in module imports:
        from dlkit.abstract_osid.id.primitives import Id as ABCId
        from dlkit.abstract_osid.type.primitives import Type as ABCType
        if not isinstance(objective_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        if not isinstance(resource_id, ABCId):
            raise errors.InvalidArgument('argument is not a valid OSID Id')
        for arg in proficiency_record_types:
            if not isinstance(arg, ABCType):
                raise errors.InvalidArgument('one or more argument array elements is not a valid OSID Type')
        if proficiency_record_types == []:
            # WHY are we passing objective_bank_id = self._catalog_id below, seems redundant:
            obj_form = objects.ProficiencyForm(
                objective_bank_id=self._catalog_id,
                objective_id=objective_id,
                resource_id=resource_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        else:
            obj_form = objects.ProficiencyForm(
                objective_bank_id=self._catalog_id,
                record_types=proficiency_record_types,
                objective_id=objective_id,
                resource_id=resource_id,
                catalog_id=self._catalog_id,
                runtime=self._runtime,
                proxy=self._proxy)
        obj_form._for_update = False
        self._forms[obj_form.get_id().get_identifier()] = not CREATED
        return obj_form