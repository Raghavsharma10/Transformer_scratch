def set_rubric(self, assessment_id):
        """Sets the rubric expressed as another assessment.

        arg:    assessment_id (osid.id.Id): the assessment ``Id``
        raise:  InvalidArgument - ``assessment_id`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``assessment_id`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.set_avatar_template
        if self.get_rubric_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_id(assessment_id):
            raise errors.InvalidArgument()
        self._my_map['rubricId'] = str(assessment_id)