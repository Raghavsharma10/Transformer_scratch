def assign_activity_to_objective_bank(self, activity_id, objective_bank_id):
        """Adds an existing ``Activity`` to a ``ObjectiveBank``.

        arg:    activity_id (osid.id.Id): the ``Id`` of the ``Activity``
        arg:    objective_bank_id (osid.id.Id): the ``Id`` of the
                ``ObjectiveBank``
        raise:  AlreadyExists - ``activity_id`` already mapped to
                ``objective_bank_id``
        raise:  NotFound - ``activity_id`` or ``objective_bank_id`` not
                found
        raise:  NullArgument - ``activity_id`` or ``objective_bank_id``
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
        self._assign_object_to_catalog(activity_id, objective_bank_id)