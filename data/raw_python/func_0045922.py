def clear_distribute_compositions(self):
        """Removes the distribution rights.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_group_template
        if (self.get_distribute_compositions_metadata().is_read_only() or
                self.get_distribute_compositions_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['distributeCompositions'] = self._distribute_compositions_default