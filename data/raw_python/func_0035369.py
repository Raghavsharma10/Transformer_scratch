def to_utf8(text):
    """ Enforce UTF8 encoding.
    """
    # return empty/false stuff unaltered
    if not text:
        if isinstance(text, string_types):
            text = ""
        return text

    try:
        # Is it a unicode string, or pure ascii?
        return text.encode("utf8")
    except UnicodeDecodeError:
        try:
            # Is it a utf8 byte string?
            if text.startswith(codecs.BOM_UTF8):
                text = text[len(codecs.BOM_UTF8):]
            return text.decode("utf8").encode("utf8")
        except UnicodeDecodeError:
            # Check BOM
            if text.startswith(codecs.BOM_UTF16_LE):
                encoding = "utf-16le"
                text = text[len(codecs.BOM_UTF16_LE):]
            elif text.startswith(codecs.BOM_UTF16_BE):
                encoding = "utf-16be"
                text = text[len(codecs.BOM_UTF16_BE):]
            else:
                # Assume CP-1252
                encoding = "cp1252"

            try:
                return text.decode(encoding).encode("utf8")
            except UnicodeDecodeError as exc:
                for line in text.splitlines():
                    try:
                        line.decode(encoding).encode("utf8")
                    except UnicodeDecodeError:
                        log.warn("Cannot transcode the following into UTF8 cause of %s: %r" % (exc, line))
                        break
                return text