def clear_branding(self):
        """Removes the branding.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if (self.get_branding_metadata().is_read_only() or
                self.get_branding_metadata().is_required()):
            raise NoAccess()
        self.my_osid_object_form._my_map['brandingIds'] = self._branding_default