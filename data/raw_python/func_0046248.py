def set_title(self, title=None):
        """Sets the title.

        :param title: the new title
        :type title: ``string``
        :raise: ``InvalidArgument`` -- ``title`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``title`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if title is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['title'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(title, metadata, array=False):
            self._my_map['title']['text'] = title
        else:
            raise InvalidArgument()