def set_published(self, published=None):
        """Sets the published status.

        :param published: the published status
        :type published: ``boolean``
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``

        *compliance: mandatory -- This method must be implemented.*

        """
        if published is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['published'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(published, metadata, array=False):
            self._my_map['published'] = published
        else:
            raise InvalidArgument()