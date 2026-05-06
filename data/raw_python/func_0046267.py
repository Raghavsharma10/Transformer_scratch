def set_url(self, url=None):
        """Sets the url.

        :param url: the new copyright
        :type url: ``string``
        :raise: ``InvalidArgument`` -- ``url`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``url`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if url is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['url'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(url, metadata, array=False):
            self._my_map['url'] = url
        else:
            raise InvalidArgument()