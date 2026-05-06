def clear_copyright_registration(self):
        """Removes the copyright registration.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetContentForm.clear_url_template
        if (self.get_copyright_registration_metadata().is_read_only() or
                self.get_copyright_registration_metadata().is_required()):
            raise errors.NoAccess()
        self._my_map['copyrightRegistration'] = self._copyright_registration_default