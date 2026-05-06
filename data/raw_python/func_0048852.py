def reassign_assessment_part_to_bank(self, assessment_part_id, from_biank_id, to_bank_id):
        """Moves an ``AssessmentPart`` from one ``Bank`` to another.

        Mappings to other ``Banks`` are unaffected.

        arg:    assessment_part_id (osid.id.Id): the ``Id`` of the
                ``AssessmentPart``
        arg:    from_biank_id (osid.id.Id): the ``Id`` of the current
                ``Bank``
        arg:    to_bank_id (osid.id.Id): the ``Id`` of the destination
                ``Bank``
        raise:  NotFound - ``assessment_part_id, from_bank_id,`` or
                ``to_bank_id`` not found or ``assessment_part_id`` not
                mapped to ``from_bank_id``
        raise:  NullArgument - ``assessment_part_id, from_bank_id,`` or
                ``to_bank_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.reassign_resource_to_bin
        self.assign_assessment_part_to_bank(assessment_part_id, to_bank_id)
        try:
            self.unassign_assessment_part_from_bank(assessment_part_id, from_biank_id)
        except:  # something went wrong, roll back assignment to to_bank_id
            self.unassign_assessment_part_from_bank(assessment_part_id, to_bank_id)
            raise