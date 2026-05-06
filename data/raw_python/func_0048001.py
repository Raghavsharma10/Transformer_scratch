def finish_assessment(self, assessment_taken_id):
        """Indicates the entire assessment is complete.

        arg:    assessment_taken_id (osid.id.Id): ``Id`` of the
                ``AssessmentTaken``
        raise:  IllegalState - ``has_begun()`` is ``false or is_over()``
                is ``true``
        raise:  NotFound - ``assessment_taken_id`` is not found
        raise:  NullArgument - ``assessment_taken_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        assessment_taken = self._get_assessment_taken(assessment_taken_id)
        assessment_taken_map = assessment_taken._my_map
        if assessment_taken.has_started() and not assessment_taken.has_ended():
            assessment_taken_map['completionTime'] = DateTime.utcnow()
            assessment_taken_map['ended'] = True
            collection = JSONClientValidated('assessment',
                                             collection='AssessmentTaken',
                                             runtime=self._runtime)
            collection.save(assessment_taken_map)
        else:
            raise errors.IllegalState()