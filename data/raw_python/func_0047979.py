def has_previous_assessment_section(self, assessment_section_id):
        """Tests if there is a previous assessment section in the assessment following the given assessment section ``Id``.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        return: (boolean) - ``true`` if there is a previous assessment
                section, ``false`` otherwise
        raise:  IllegalState - ``has_assessment_begun()`` is ``false``
        raise:  NotFound - ``assessment_section_id`` is not found
        raise:  NullArgument - ``assessment_section_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        try:
            self.get_previous_assessment_section(assessment_section_id)
        except errors.IllegalState:
            return False
        else:
            return True