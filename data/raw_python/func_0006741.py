def _load_hangul_syllable_types():
    """
    Helper function for parsing the contents of "HangulSyllableType.txt" from the Unicode Character Database (UCD) and
    generating a lookup table for determining whether or not a given Hangul syllable is of type "L", "V", "T", "LV" or
    "LVT".  For more info on the UCD, see the following website: https://www.unicode.org/ucd/
    """
    filename = "HangulSyllableType.txt"
    current_dir = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(current_dir, filename), mode="r", encoding="utf-8") as fp:
        for line in fp:
            if not line.strip() or line.startswith("#"):
                continue  # Skip empty lines or lines that are comments (comments start with '#')
            data = line.strip().split(";")
            syllable_type, _ = map(six.text_type.strip, data[1].split("#"))
            if ".." in data[0]:  # If it is a range and not a single value
                start, end = map(lambda x: int(x, 16), data[0].strip().split(".."))
                for idx in range(start, end + 1):
                    _hangul_syllable_types[idx] = syllable_type
            else:
                _hangul_syllable_types[int(data[0].strip(), 16)] = syllable_type