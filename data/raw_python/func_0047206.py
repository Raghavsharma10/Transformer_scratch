def unassign_proficiency_from_objective_bank(self, proficiency_id, objective_bank_id):
        """Removes a ``Proficiency`` from a ``ObjectiveBank``.

        arg:    proficiency_id (osid.id.Id): the ``Id`` of the
                ``Proficiency``
        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                ``ObjectiveBank``
        raise:  NotFound - ``proficiency_id`` or ``objective_bank_id``
                not found or ``proficiency_id`` not mapped to
                ``objective_bank_id``
        raise:  NullArgument - ``proficiency_id`` or
                ``objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('LEARNING', local=True)
        lookup_session = mgr.get_objective_bank_lookup_session(proxy=self._proxy)
        lookup_session.get_objective_bank(objective_bank_id)  # to raise NotFound
        self._unassign_object_from_catalog(proficiency_id, objective_bank_id)