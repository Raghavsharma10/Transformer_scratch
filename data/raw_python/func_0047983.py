def has_assessment_section_begun(self, assessment_section_id):
        """Tests if this assessment section has started.

        A section begins from the designated start time if a start time
        is defined. If no start time is defined the section may begin at
        any time. Assessment items cannot be accessed or submitted if
        the return for this method is ``false``.

        arg:    assessment_section_id (osid.id.Id): ``Id`` of the
                ``AssessmentSection``
        return: (boolean) - ``true`` if this assessment section has
                begun, ``false`` otherwise
        raise:  IllegalState - ``has_assessment_begun()`` is ``false or
                is_assessment_over()`` is ``true``
        raise:  NotFound - ``assessment_section_id`` is not found
        raise:  NullArgument - ``assessment_section_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        return get_section_util(assessment_section_id,
                                runtime=self._runtime)._assessment_taken.has_started()