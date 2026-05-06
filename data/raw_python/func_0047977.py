def has_next_assessment_section(self, assessment_section_id):
        """Tests if there is a next assessment section in the assessment following the given assessment section ``Id``.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        return: (boolean) - ``true`` if there is a next section,
                ``false`` otherwise
        raise:  IllegalState - ``has_assessment_begun()`` is ``false``
        raise:  NotFound - ``assessment_taken_id`` is not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure occurred
        *compliance: mandatory -- This method must be implemented.*

        """
        try:
            self.get_next_assessment_section(assessment_section_id)
        except errors.IllegalState:
            return False
        else:
            return True