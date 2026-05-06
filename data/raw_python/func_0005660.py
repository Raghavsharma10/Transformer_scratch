def text_badness(text):
    u'''
    Look for red flags that text is encoded incorrectly:

    Obvious problems:
    - The replacement character \ufffd, indicating a decoding error
    - Unassigned or private-use Unicode characters

    Very weird things:
    - Adjacent letters from two different scripts
    - Letters in scripts that are very rarely used on computers (and
      therefore, someone who is using them will probably get Unicode right)
    - Improbable control characters, such as 0x81

    Moderately weird things:
    - Improbable single-byte characters, such as ƒ or ¬
    - Letters in somewhat rare scripts
    '''
    assert isinstance(text, str)
    errors = 0
    very_weird_things = 0
    weird_things = 0
    prev_letter_script = None
    unicodedata_name = unicodedata.name
    unicodedata_category = unicodedata.category
    for char in text:
        index = ord(char)
        if index < 256:
            # Deal quickly with the first 256 characters.
            weird_things += SINGLE_BYTE_WEIRDNESS[index]
            if SINGLE_BYTE_LETTERS[index]:
                prev_letter_script = 'latin'
            else:
                prev_letter_script = None
        else:
            category = unicodedata_category(char)
            if category == 'Co':
                # Unassigned or private use
                errors += 1
            elif index == 0xfffd:
                # Replacement character
                errors += 1
            elif index in WINDOWS_1252_GREMLINS:
                lowchar = char.encode('WINDOWS_1252').decode('latin-1')
                weird_things += SINGLE_BYTE_WEIRDNESS[ord(lowchar)] - 0.5

            if category[0] == 'L':
                # It's a letter. What kind of letter? This is typically found
                # in the first word of the letter's Unicode name.
                name = unicodedata_name(char)
                scriptname = name.split()[0]
                freq, script = SCRIPT_TABLE.get(scriptname, (0, 'other'))
                if prev_letter_script:
                    if script != prev_letter_script:
                        very_weird_things += 1
                    if freq == 1:
                        weird_things += 2
                    elif freq == 0:
                        very_weird_things += 1
                prev_letter_script = script
            else:
                prev_letter_script = None

    return 100 * errors + 10 * very_weird_things + weird_things