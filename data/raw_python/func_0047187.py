def reassign_activity_to_objective_bank(self, activity_id, from_objective_bank_id, to_objective_bank_id):
        """Moves an ``Activity`` from one ``ObjectiveBank`` to another.

        Mappings to other ``ObjectiveBanks`` are unaffected.

        arg:    activity_id (osid.id.Id): the ``Id`` of the ``Activity``
        arg:    from_objective_bank_id (osid.id.Id): the ``Id`` of the
                current ``ObjectiveBank``
        arg:    to_objective_bank_id (osid.id.Id): the ``Id`` of the
                destination ``ObjectiveBank``
        raise:  NotFound - ``activity_id, from_objective_bank_id,`` or
                ``to_objective_bank_id`` not found or ``activity_id``
                not mapped to ``from_objective_bank_id``
        raise:  NullArgument - ``activity_id, from_objective_bank_id,``
                or ``to_objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.reassign_resource_to_bin
        self.assign_activity_to_objective_bank(activity_id, to_objective_bank_id)
        try:
            self.unassign_activity_from_objective_bank(activity_id, from_objective_bank_id)
        except:  # something went wrong, roll back assignment to to_objective_bank_id
            self.unassign_activity_from_objective_bank(activity_id, to_objective_bank_id)
            raise