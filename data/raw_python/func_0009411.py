def encoding_for(source_path, encoding='automatic', fallback_encoding=None):
    """
    The encoding used by the text file stored in ``source_path``.

    The algorithm used is:

    * If ``encoding`` is ``'automatic``, attempt the following:
      1. Check BOM for UTF-8, UTF-16 and UTF-32.
      2. Look for XML prolog or magic heading like ``# -*- coding: cp1252 -*-``
      3. Read the file using UTF-8.
      4. If all this fails, use assume the ``fallback_encoding``.
    * If ``encoding`` is ``'chardet`` use :mod:`chardet` to obtain the encoding.
    * For any other ``encoding`` simply use the specified value.
    """
    assert encoding is not None

    if encoding == 'automatic':
        with open(source_path, 'rb') as source_file:
            heading = source_file.read(128)
        result = None
        if len(heading) == 0:
            # File is empty, assume a dummy encoding.
            result = 'utf-8'
        if result is None:
            # Check for known BOMs.
            for bom, encoding in _BOM_TO_ENCODING_MAP.items():
                if heading[:len(bom)] == bom:
                    result = encoding
                    break
        if result is None:
            # Look for common headings that indicate the encoding.
            ascii_heading = heading.decode('ascii', errors='replace')
            ascii_heading = ascii_heading.replace('\r\n', '\n')
            ascii_heading = ascii_heading.replace('\r', '\n')
            ascii_heading = '\n'.join(ascii_heading.split('\n')[:2]) + '\n'
            coding_magic_match = _CODING_MAGIC_REGEX.match(ascii_heading)
            if coding_magic_match is not None:
                result = coding_magic_match.group('encoding')
            else:
                first_line = ascii_heading.split('\n')[0]
                xml_prolog_match = _XML_PROLOG_REGEX.match(first_line)
                if xml_prolog_match is not None:
                    result = xml_prolog_match.group('encoding')
    elif encoding == 'chardet':
        assert _detector is not None, \
            'without chardet installed, encoding="chardet" must be rejected before calling encoding_for()'
        _detector.reset()
        with open(source_path, 'rb') as source_file:
            for line in source_file.readlines():
                _detector.feed(line)
                if _detector.done:
                    break
        result = _detector.result['encoding']
        if result is None:
            _log.warning(
                '%s: chardet cannot determine encoding, assuming fallback encoding %s',
                source_path, fallback_encoding)
            result = fallback_encoding
    else:
        # Simply use the specified encoding.
        result = encoding
    if result is None:
        # Encoding 'automatic' or 'chardet' failed to detect anything.
        if fallback_encoding is not None:
            # If defined, use the fallback encoding.
            result = fallback_encoding
        else:
            try:
                # Attempt to read the file as UTF-8.
                with open(source_path, 'r', encoding='utf-8') as source_file:
                    source_file.read()
                result = 'utf-8'
            except UnicodeDecodeError:
                # UTF-8 did not work out, use the default as last resort.
                result = DEFAULT_FALLBACK_ENCODING
            _log.debug('%s: no fallback encoding specified, using %s', source_path, result)

    assert result is not None
    return result