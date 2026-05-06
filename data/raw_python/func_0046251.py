def set_copyright(self, copyright=None):
        """Sets the copyright.

        :param copyright: the new copyright
        :type copyright: ``string``
        :raise: ``InvalidArgument`` -- ``copyright`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``copyright`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if copyright is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['copyright'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(copyright, metadata, array=False):
            self._my_map['copyright']['text'] = copyright
        else:
            raise InvalidArgument()