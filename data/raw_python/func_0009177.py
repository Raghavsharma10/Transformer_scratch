def _convert(s, re_pattern, syllable_function, add_apostrophes=False,
             remove_apostrophes=False, separate_syllables=False):
    """Convert a string's syllables to a different transcription system."""
    original = s
    new = ''
    while original:
        match = re.search(re_pattern, original, re.IGNORECASE | re.UNICODE)
        if match is None and original:
            # There are no more matches, but the given string isn't fully
            # processed yet.
            new += original
            break
        match_start, match_end = match.span()
        if match_start > 0:  # Handle extra characters before matched syllable.
            if (new and remove_apostrophes and match_start == 1 and
                    original[0] == "'"):
                pass  # Remove the apostrophe between Pinyin syllables.
                if separate_syllables:  # Separate syllables by a space.
                    new += ' '
            else:
                new += original[0:match_start]
        else:  # Matched syllable starts immediately.
            if new and separate_syllables:  # Separate syllables by a space.
                new += ' '
            elif (new and add_apostrophes and
                    match.group()[0].lower() in _UNACCENTED_VOWELS):
                new += "'"
        # Convert the matched syllable.
        new += syllable_function(match.group())
        original = original[match_end:]
    return new