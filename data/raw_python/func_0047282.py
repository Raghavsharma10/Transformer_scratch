def get_child_assessment_parts(self):
        """Gets any child assessment parts.

        return: (osid.assessment.authoring.AssessmentPartList) - the
                child assessment parts
        raise:  OperationFailed - unable to complete request
        *compliance: mandatory -- This method must be implemented.*

        """
        if not self.has_children():
            raise errors.IllegalState('no children assessment parts')
        # only returned unsequestered children?
        lookup_session = self._get_assessment_part_lookup_session()
        lookup_session.use_sequestered_assessment_part_view()
        return lookup_session.get_assessment_parts_by_ids(self.get_child_ids())