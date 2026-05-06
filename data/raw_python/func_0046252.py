def set_copyright_registration(self, registration):
        """Sets the copyright registration.

        arg:    registration (string): the new copyright registration
        raise:  InvalidArgument - ``copyright`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``copyright`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for osid.repository.AssetContentForm.set_url_template
        if self.get_copyright_registration_metadata().is_read_only():
            raise NoAccess()
        if not self._is_valid_string(
                registration,
                self.get_copyright_registration_metadata()):
            raise InvalidArgument()
        self._my_map['copyrightRegistration'] = registration