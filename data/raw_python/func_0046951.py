def set_assessment(self, assessment_id):
        """Sets the assessment.

        arg:    assessment_id (osid.id.Id): the new assessment
        raise:  InvalidArgument - ``assessment_id`` is invalid
        raise:  NoAccess - ``assessment_id`` cannot be modified
        raise:  NullArgument - ``assessment_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_assessment_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(assessment_id):
            raise errors.InvalidArgument()
        self._my_map['assessmentId'] = str(assessment_id)