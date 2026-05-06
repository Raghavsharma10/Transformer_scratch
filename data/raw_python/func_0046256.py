def set_source(self, source_id=None):
        """Sets the source.

        :param source_id: the new publisher
        :type source_id: ``osid.id.Id``
        :raise: ``InvalidArgument`` -- ``source_id`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``source_id`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if source_id is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['source_id'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(source_id, metadata, array=False):
            self._my_map['sourceId'] = str(source_id)
        else:
            raise InvalidArgument()