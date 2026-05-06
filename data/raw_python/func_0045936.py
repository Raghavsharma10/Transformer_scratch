def set_published_date(self, published_date):
        """Sets the published date.

        arg:    published_date (osid.calendaring.DateTime): the new
                published date
        raise:  InvalidArgument - ``published_date`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``published_date`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.assessment.AssessmentOfferedForm.set_start_time_template
        if self.get_published_date_metadata().is_read_only():
            raise errors.NoAccess()
        if not self._is_valid_date_time(
                published_date,
                self.get_published_date_metadata()):
            raise errors.InvalidArgument()
        self._my_map['publishedDate'] = published_date