def _load_data(self):
        """
        Load the ARPABET ASCII IPA data from the built-in database.
        """
        ipa_canonical_string_to_ascii_str = dict()
        for line in load_data_file(
            file_path=self.DATA_FILE_PATH,
            file_path_is_relative=True,
            line_format=u"UA"
        ):
            i_unicode, i_ascii = line
            if (len(i_unicode) == 0) or (len(i_ascii) == 0):
                raise ValueError("Data file '%s' contains a bad line: '%s'" % (self.DATA_FILE_PATH, line))
            i_unicode = i_unicode[0]
            i_ascii = i_ascii[0]
            key = tuple([UNICODE_TO_IPA[c].canonical_representation for c in i_unicode])
            ipa_canonical_string_to_ascii_str[key] = i_ascii
        return ipa_canonical_string_to_ascii_str