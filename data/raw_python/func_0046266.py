def remove_accessibility_type(self, accessibility_type=None):
        """Removes an accessibility type.

        :param accessibility_type: accessibility type to remove
        :type accessibility_type: ``osid.type.Type``
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NotFound`` -- acessibility type not found
        :raise: ``NullArgument`` -- ``accessibility_type`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if accessibility_type is None:
            raise NullArgument
        metadata = Metadata(**settings.METADATA['accessibility_type'])
        if metadata.is_read_only() or metadata.is_required():
            raise NoAccess()
        if (accessibility_type._my_map['id']) not in self._my_map['accessibility_type']:
            raise NotFound()
        self._my_map['accessibility_types'].remove(accessibility_type._my_map['id'])