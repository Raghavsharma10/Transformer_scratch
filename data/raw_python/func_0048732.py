def add_transcript_file(self, transcript_file, language_type=None):
        """Adds a transcript file tagged as the given language.

        arg:    transcript_file (displayText): the new transcript_file
        raise:  InvalidArgument - ``transcript_file`` is invalid
        raise:  NoAccess - ``Metadata.isReadOnly()`` is ``true``
        raise:  NullArgument - ``media_description`` is ``null``
        *compliance: mandatory -- This method must be implemented.*

        """
        if not isinstance(transcript_file, DataInputStream):
            raise InvalidArgument('transcript_file')
        # for now, don't bother with genusTypeIds for the newly created
        # asset or assetContent...supposed to be managed via this one, I think
        locale = DEFAULT_LANGUAGE_TYPE.identifier
        if language_type is not None:
            locale = language_type.identifier
        self.my_osid_object_form.add_file(transcript_file,
                                          locale,
                                          asset_name="Transcript File Container",
                                          asset_description="Used by an asset content to manage multi-language Transcript files")