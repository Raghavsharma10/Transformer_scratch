def clear_assessments(self):
        """Clears the assessments.

        raise:  NoAccess - ``Metadata.isRequired()`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.learning.ActivityForm.clear_assets_template
        if (self.get_assessments_metadata().is_read_only() or
                self.get_assessments_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['assessmentIds'] = self._assessments_default