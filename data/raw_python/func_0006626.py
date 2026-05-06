def _build_unicode_character_database(self):
        """
        Function for parsing the Unicode character data from the Unicode Character
        Database (UCD) and generating a lookup table.  For more info on the UCD,
        see the following website: https://www.unicode.org/ucd/
        """
        filename = "UnicodeData.txt"
        current_dir = os.path.abspath(os.path.dirname(__file__))
        tag = re.compile(r"<\w+?>")
        with codecs.open(os.path.join(current_dir, filename), mode="r", encoding="utf-8") as fp:
            for line in fp:
                if not line.strip():
                    continue
                data = line.strip().split(";")
                # Replace the start/end range markers with their proper derived names.
                if data[1].endswith((u"First>", u"Last>")) and _is_derived(int(data[0], 16)):
                    data[1] = _get_nr_prefix(int(data[0], 16))
                    if data[1].startswith("HANGUL SYLLABLE"):  # For Hangul syllables, use naming rule NR1
                        data[1] += _get_hangul_syllable_name(int(data[0], 16))
                    else:  # Others should use naming rule NR2
                        data[1] += data[0]
                data[3] = int(data[3])  # Convert the Canonical Combining Class value into an int.
                if data[5]:  # Convert the contents of the decomposition into characters, preserving tag info.
                    data[5] = u" ".join([_hexstr_to_unichr(s) if not tag.match(s) else s for s in data[5].split()])
                for i in [6, 7, 8]:  # Convert the decimal, digit and numeric fields to either ints or fractions.
                    if data[i]:
                        if "/" in data[i]:
                            data[i] = Fraction(data[i])
                        else:
                            data[i] = int(data[i])
                for i in [12, 13, 14]:  # Convert the uppercase, lowercase and titlecase fields to characters.
                    if data[i]:
                        data[i] = _hexstr_to_unichr(data[i])
                lookup_name = _uax44lm2transform(data[1])
                uc_data = UnicodeCharacter(u"U+" + data[0], *data[1:])
                self._unicode_character_database[int(data[0], 16)] = uc_data
                self._name_database[lookup_name] = uc_data
        # Fill out the "compressed" ranges of UnicodeData.txt i.e. fill out the remaining characters per the Name
        # Derivation Rules.  See the Unicode Standard, ch. 4, section 4.8, Unicode Name Property
        for lookup_range, prefix_string in _nr_prefix_strings.items():
            exemplar = self._unicode_character_database.__getitem__(lookup_range[0])
            for item in lookup_range:
                hex_code = _padded_hex(item)
                new_name = prefix_string
                if prefix_string.startswith("HANGUL SYLLABLE"):  # For Hangul, use naming rule NR1
                    new_name += _get_hangul_syllable_name(item)
                else:  # Everything else uses naming rule NR2
                    new_name += hex_code
                uc_data = exemplar._replace(code=u"U+" + hex_code, name=new_name)
                self._unicode_character_database[item] = uc_data
                self._name_database[_uax44lm2transform(new_name)] = uc_data