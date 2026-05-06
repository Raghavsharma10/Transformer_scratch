def set_published_date(self, published_date=None):
        """Sets the published date.

        :param published_date: the new published date
        :type published_date: ``osid.calendaring.DateTime``
        :raise: ``InvalidArgument`` -- ``published_date`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``published_date`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if published_date is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['published_date'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(published_date, metadata, array=False):
            self._my_map['publishedDate'] = published_date  # This is probably wrong
        else:
            raise InvalidArgument()