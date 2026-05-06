def _load_data(self):
        """
        Load the Kirshenbaum ASCII IPA data from the built-in database.
        """
        ipa_canonical_string_to_ascii_str = dict()
        for line in load_data_file(
            file_path=self.DATA_FILE_PATH,
            file_path_is_relative=True,
            line_format=u"sxA"
        ):
            i_desc, i_ascii = line
            if len(i_ascii) == 0:
                raise ValueError("Data file '%s' contains a bad line: '%s'" % (self.DATA_FILE_PATH, line))
            key = (variant_to_canonical_string(i_desc),)
            ipa_canonical_string_to_ascii_str[key] = i_ascii[0]
        return ipa_canonical_string_to_ascii_str