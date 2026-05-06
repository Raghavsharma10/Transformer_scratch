def set_license(self, license_):
        """Sets the license.

        arg:    license (string): the new license
        raise:  InvalidArgument - ``license`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``license`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if license_ is None:
            raise NullArgument('license cannot be None')
        if not utilities.is_string(license_):
            raise InvalidArgument('license must be a string')
        if self.get_license_metadata().is_read_only():
            raise NoAccess()
        if not self.my_osid_object_form._is_valid_string(license_, self.get_license_metadata()):
            raise InvalidArgument()
        self.my_osid_object_form._my_map['license']['text'] = license_