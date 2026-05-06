def reassign_proficiency_to_objective_bank(self, objective_id, from_objective_bank_id, to_objective_bank_id):
        """Moves an ``Objective`` from one ``ObjectiveBank`` to another.

        Mappings to other ``ObjectiveBanks`` are unaffected.

        arg:    objective_id (osid.id.Id): the ``Id`` of the
                ``Objective``
        arg:    from_objective_bank_id (osid.id.Id): the ``Id`` of the
                current ``ObjectiveBank``
        arg:    to_objective_bank_id (osid.id.Id): the ``Id`` of the
                destination ``ObjectiveBank``
        raise:  NotFound - ``objective_id, from_objective_bank_id,`` or
                ``to_objective_bank_id`` not found or ``objective_id``
                not mapped to ``from_objective_bank_id``
        raise:  NullArgument - ``objective_id, from_objective_bank_id,``
                or ``to_objective_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.reassign_resource_to_bin
        self.assign_objective_to_objective_bank(objective_id, to_objective_bank_id)
        try:
            self.unassign_objective_from_objective_bank(objective_id, from_objective_bank_id)
        except:  # something went wrong, roll back assignment to to_objective_bank_id
            self.unassign_objective_from_objective_bank(objective_id, to_objective_bank_id)
            raise