def remove_display_name_by_language(self, language_type):
        """Removes the specified display_name.

        raise:  NoAccess - ``Metadata.isRequired()`` is ``true`` or
                ``Metadata.isReadOnly()`` is ``true``
        *compliance: mandatory -- This method must be implemented.*

        """
        if self.get_display_names_metadata().is_read_only():
            raise NoAccess()
        if not isinstance(language_type, Type):
            raise InvalidArgument('language_type must be instance of Type')
        self.my_osid_object_form._my_map['displayNames'] = [t
                                                            for t in self.my_osid_object_form._my_map['displayNames']
                                                            if t['languageTypeId'] != str(language_type)]