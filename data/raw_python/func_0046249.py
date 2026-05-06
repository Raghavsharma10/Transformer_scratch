def clear_title(self):
        """Removes the title.

        :raise: ``NoAccess`` -- ``Metadata.isRequired()`` is ``true`` or
            ``Metadata.isReadOnly()`` is ``true``

        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = Metadata(**settings.METADATA['title'])
        if metadata.is_read_only() or metadata.is_required():
            raise NoAccess()
        self._my_map['title']['text'] = ''