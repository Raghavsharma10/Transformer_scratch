def set_created_date(self, created_date=None):
        """Sets the created date.

        :param created_date: the new created date
        :type created_date: ``osid.calendaring.DateTime``
        :raise: ``InvalidArgument`` -- ``created_date`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``created_date`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if created_date is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['created_date'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(created_date, metadata, array=False):
            self._my_map['createdDate'] = created_date  # This is probably wrong
        else:
            raise InvalidArgument()