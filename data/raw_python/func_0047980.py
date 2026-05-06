def get_previous_assessment_section(self, assessment_section_id):
        """Gets the next assessemnt section following the given assesment section.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        return: (osid.assessment.AssessmentSection) - the previous
                assessment section
        raise:  IllegalState - ``has_next_assessment_section()`` is
                ``false``
        raise:  NotFound - ``assessment_section_id`` is not found
        raise:  NullArgument - ``assessment_section_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        assessment_taken = self.get_assessment_section(assessment_section_id)._assessment_taken
        return assessment_taken._get_previous_assessment_section(assessment_section_id)