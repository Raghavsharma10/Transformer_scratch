def clear_composition(self):
        """Removes the composition link.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_avatar_template
        if (self.get_composition_metadata().is_read_only() or
                self.get_composition_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['compositionId'] = self._composition_default