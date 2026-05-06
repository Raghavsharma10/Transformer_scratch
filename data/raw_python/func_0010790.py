def replace_html_entities(cls, replacement_string):
        """Replaces HTML escaped unicode entities with their unicode
        equivalent. If the setting `EMOJI_REPLACE_HTML_ENTITIES` is
        `True` then this conversation will always be done in
        `replace_unicode` (default: True).

        """
        def _hex_to_unicode(hex_code):
            if PYTHON3:
                hex_code = '{0:0>8}'.format(hex_code)
                as_int = struct.unpack('>i', bytes.fromhex(hex_code))[0]
                return '{0:c}'.format(as_int)
            else:
                return hex_to_unicode(hex_code)

        def _replace_integer_entity(match):
            hex_val = hex(int(match.group(1)))

            return _hex_to_unicode(hex_val.replace('0x', ''))

        def _replace_hex_entity(match):
            return _hex_to_unicode(match.group(1))

        # replace integer code points, &#65;
        replacement_string = re.sub(
            cls._html_entities_integer_unicode_regex,
            _replace_integer_entity,
            replacement_string
        )
        # replace hex code points, &#x41;
        replacement_string = re.sub(
            cls._html_entities_hex_unicode_regex,
            _replace_hex_entity,
            replacement_string
        )

        return replacement_string