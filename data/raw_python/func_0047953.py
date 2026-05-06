def set_genus_type(self, genus_type=None):
        """Sets a genus.

        A genus cannot be cleared because all objects have at minimum a
        root genus.

        arg:    genusType (osid.type.Type): the new genus
        raise:  InvalidArgument - genusType is invalid
        raise:  NoAccess - metadata.is_readonly() is true
        raise:  NullArgument - genusType is null
        compliance: mandatory - This method must be implemented.

        """
        if genus_type is None:
            raise NullArgument()
        metadata = Metadata(**settings.METADATA['genus_type'])
        metadata_id = Metadata(**settings.METADATA['genus_type_id'])
        if metadata.is_read_only():
            raise NoAccess()
        if self._is_valid_input(genus_type, metadata, array=False):
            self._my_map['genusTypeId'] = str(genus_type)
            # REALLY?  This assumes that all genus_type arguments
            # will be Types that have come from Hancar.  Perhaps?
        elif self._is_valid_input(genus_type, metadata_id, array=False):
            self._my_map['genusTypeId'] = str(genus_type)
        else:
            raise InvalidArgument