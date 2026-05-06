def replace_unicode(cls, replacement_string):
        """This method will iterate over every character in
        ``replacement_string`` and see if it mathces any of the
        unicode codepoints that we recognize. If it does then it will
        replace that codepoint with an image just like ``replace``.

        NOTE: This will only work with Python versions built with wide
        unicode caracter support. Python 3 should always work but
        Python 2 will have to tested before deploy.

        """
        e = cls()
        output = []
        surrogate_character = None

        if settings.EMOJI_REPLACE_HTML_ENTITIES:
            replacement_string = cls.replace_html_entities(replacement_string)

        for i, character in enumerate(replacement_string):
            if character in cls._unicode_modifiers:
                continue

            # Check whether this is the first character in a Unicode
            # surrogate pair when Python doesn't have wide Unicode
            # support.
            #
            # Is there any reason to do this even if Python got wide
            # support enabled?
            if(not UNICODE_WIDE and not surrogate_character and
               ord(character) >= UNICODE_SURROGATE_MIN and
               ord(character) <= UNICODE_SURROGATE_MAX):
                surrogate_character = character
                continue

            if surrogate_character:
                character = convert_unicode_surrogates(
                    surrogate_character + character
                )
                surrogate_character = None

            name = e.name_for(character)
            if name:
                if settings.EMOJI_ALT_AS_UNICODE:
                    character = e._image_string(name, alt=character)
                else:
                    character = e._image_string(name)

            output.append(character)

        return ''.join(output)