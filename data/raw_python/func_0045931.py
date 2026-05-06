def clear_created_date(self):
        """Removes the created date.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.clear_start_time_template
        if (self.get_created_date_metadata().is_read_only() or
                self.get_created_date_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['createdDate'] = self._created_date_default