def add_accessibility_type(self, accessibility_type=None):
        """Adds an accessibility type.

        Multiple types can be added.

        :param accessibility_type: a new accessibility type
        :type accessibility_type: ``osid.type.Type``
        :raise: ``InvalidArgument`` -- ``accessibility_type`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``accessibility_t_ype`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if accessibility_type is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['accessibility_type'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(accessibility_type, metadata, array=False):
            self._my_map['accessibilityTypeIds'].append(accessibility_type._my_map['id'])
            # REALLY?  This assumes that all accessibility_type arguments
            # will be Types that have come from Handcar.  Perhaps?
        else:
            raise InvalidArgument