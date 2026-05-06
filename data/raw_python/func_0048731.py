def add_vtt_file(self, vtt_file, language_type=None):
        """Adds a vtt file tagged as the given language.

        arg:    vtt_file (displayText): the new vtt_file
        raise:  InvalidArgument - ``vtt_file`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``media_description`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(vtt_file, DataInputStream):
            raise InvalidArgument('vtt_file')
        # for now, don't bother with genusTypeIds for the newly created
        # asset or assetContent...supposed to be managed via this one, I think
        locale = DEFAULT_LANGUAGE_TYPE.identifier
        if language_type is not None:
            locale = language_type.identifier
        self.my_osid_object_form.add_file(vtt_file,
                                          locale,
                                          asset_name="VTT File Container",
                                          asset_description="Used by an asset content to manage multi-language VTT files")