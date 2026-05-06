def unassign_objective_requisite(self, objective_id, requisite_objective_id):
        """Removes an ``Objective`` requisite from an ``Objective``.

        arg:    objective_id (osid.id.Id): the ``Id`` of the
                ``Objective``
        arg:    requisite_objective_id (osid.id.Id): the ``Id`` of the
                required ``Objective``
        raise:  NotFound - ``objective_id`` or
                ``requisite_objective_id`` not found or ``objective_id``
                not mapped to ``requisite_objective_id``
        raise:  NullArgument - ``objective_id`` or
                ``requisite_objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        requisite_type = Type(**Relationship().get_type_data('OBJECTIVE.REQUISITE'))
        rls = self._get_provider_manager(
            'RELATIONSHIP').get_relationship_lookup_session_for_family(
            self.get_objective_bank_id(), proxy=self._proxy)
        ras = self._get_provider_manager(
            'RELATIONSHIP').get_relationship_admin_session_for_family(
            self.get_objective_bank_id(), proxy=self._proxy)
        rls.use_federated_family_view()
        relationships = rls.get_relationships_by_genus_type_for_source(objective_id, requisite_type)
        if relationships.available() == 0:
            raise errors.IllegalState('no Objective found')
        for relationship in relationships:
            if str(relationship.get_destination_id()) == str(requisite_objective_id):
                ras.delete_relationship(relationship.ident)