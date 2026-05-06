def assign_objective_to_objective_bank(self, objective_id, objective_bank_id):
        """Adds an existing ``Objective`` to an ``ObjectiveBank``.

        arg:    objective_id (osid.id.Id): the ``Id`` of the
                ``Objective``
        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                ``ObjectiveBank``
        raise:  AlreadyExists - ``objective_id`` already mapped to
                ``objective_bank_id``
        raise:  NotFound - ``objective_id`` or ``objective_bank_id`` not
                found
        raise:  NullArgument - ``objective_id`` or ``objective_bank_id``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.assign_resource_to_bin
        mgr = self._get_provider_manager('LEARNING', local=True)
        lookup_session = mgr.get_objective_bank_lookup_session(proxy=self._proxy)
        lookup_session.get_objective_bank(objective_bank_id)  # to raise NotFound
        self._assign_object_to_catalog(objective_id, objective_bank_id)