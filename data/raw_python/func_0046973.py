def set_assessments(self, assessment_ids):
        """Sets the assessments.

        arg:    assessment_ids (osid.id.Id[]): the assessment ``Ids``
        raise:  InvalidArgument - ``assessment_ids`` is invalid
        raise:  NullArgument - ``assessment_ids`` is ``null``
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.set_assets_template
        if not isinstance(assessment_ids, list):
            raise errors.InvalidArgument()
        if self.get_assessments_metadata().is_read_only():
            raise errors.NoAccess()
        idstr_list = []
        for object_id in assessment_ids:
            if not self._is_valid_id(object_id):
                raise errors.InvalidArgument()
            idstr_list.append(str(object_id))
        self._my_map['assessmentIds'] = idstr_list