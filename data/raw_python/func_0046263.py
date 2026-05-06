def set_composition(self, composition_id=None):
        """Sets the composition.

        :param composition_id: a composition
        :type composition_id: ``osid.id.Id``
        :raise: ``InvalidArgument`` -- ``composition_id`` is invalid
        :raise: ``NoAccess`` -- ``Metadata.isReadOnly()`` is ``true``
        :raise: ``NullArgument`` -- ``composition_id`` is ``null``

        *compliance: mandatory -- This method must be implemented.*

        """
        if composition_id is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['composition_id'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(composition_id, metadata, array=False):
            self._my_map['compositionId'] = str(composition_id)
        else:
            raise InvalidArgument()