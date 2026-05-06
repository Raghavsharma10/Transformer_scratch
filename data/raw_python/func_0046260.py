def clear_published(self):
        """Removes the published status.

        :raise: ``NoAccess`` -- ``Metadata.isRequired()`` is ``true`` or ``Metadata.isReadOnly()`` is ``true``

        *compliance: mandatory -- This method must be implemented.*

        """
        metadata = Metadata(**settings.METADATA['published'])
        if metadata.is_read_only() or metadata.is_required():
            raise NoAccess()
        self._my_map['published'] = False