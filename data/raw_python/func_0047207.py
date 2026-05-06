def reassign_proficiency_to_objective_bank(self, proficiency_id, from_objective_bank_id, to_objective_bank_id):
        """Moves a ``Proficiency`` from one ``ObjectiveBank`` to another.

        Mappings to other ``ObjectiveBanks`` are unaffected.

        arg:    proficiency_id (osid.id.Id): the ``Id`` of the
                ``Proficiency``
        arg:    from_objective_bank_id (osid.id.Id): the ``Id`` of the
                current ``ObjectiveBank``
        arg:    to_objective_bank_id (osid.id.Id): the ``Id`` of the
                destination ``ObjectiveBank``
        raise:  NotFound - ``proficiency_id, from_objective_bank_id,``
                or ``to_objective_bank_id`` not found or
                ``proficiency_id`` not mapped to
                ``from_objective_bank_id``
        raise:  NullArgument - ``proficiency_id,
                from_objective_bank_id,`` or ``to_objective_bank_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.reassign_resource_to_bin
        self.assign_proficiency_to_objective_bank(proficiency_id, to_objective_bank_id)
        try:
            self.unassign_proficiency_from_objective_bank(proficiency_id, from_objective_bank_id)
        except:  # something went wrong, roll back assignment to to_objective_bank_id
            self.unassign_proficiency_from_objective_bank(proficiency_id, to_objective_bank_id)
            raise