def clear_distribute_verbatim(self):
        """Removes the distribution rights.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.resource.ResourceForm.clear_group_template
        if (self.get_distribute_verbatim_metadata().is_read_only() or
                self.get_distribute_verbatim_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['distributeVerbatim'] = self._distribute_verbatim_default