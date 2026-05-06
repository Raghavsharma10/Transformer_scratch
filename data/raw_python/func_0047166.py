def assign_objective_requisite(self, objective_id, requisite_objective_id):
        """Creates a requirement dependency between two ``Objectives``.

        arg:    objective_id (osid.id.Id): the ``Id`` of the dependent
                ``Objective``
        arg:    requisite_objective_id (osid.id.Id): the ``Id`` of the
                required ``Objective``
        raise:  AlreadyExists - ``objective_id`` already mapped to
                ``requisite_objective_id``
        raise:  NotFound - ``objective_id`` or
                ``requisite_objective_id`` not found
        raise:  NullArgument - ``objective_id`` or
                ``requisite_objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        requisite_type = Type(**Relationship().get_type_data('OBJECTIVE.REQUISITE'))

        ras = self._get_provider_manager(
            'RELATIONSHIP').get_relationship_admin_session_for_family(
            self.get_objective_bank_id(), proxy=self._proxy)
        rfc = ras.get_relationship_form_for_create(objective_id, requisite_objective_id, [])
        rfc.set_display_name('Objective Requisite')
        rfc.set_description('An Objective Requisite created by the ObjectiveRequisiteAssignmentSession')
        rfc.set_genus_type(requisite_type)
        ras.create_relationship(rfc)