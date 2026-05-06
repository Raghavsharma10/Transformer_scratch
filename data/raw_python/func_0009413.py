def source_analysis(
        source_path, group, encoding='automatic', fallback_encoding='cp1252',
        generated_regexes=pygount.common.regexes_from(DEFAULT_GENERATED_PATTERNS_TEXT),
        duplicate_pool=None):
    """
    Analysis for line counts in source code stored in ``source_path``.

    :param source_path:
    :param group: name of a logical group the sourc code belongs to, e.g. a
      package.
    :param encoding: encoding according to :func:`encoding_for`
    :param fallback_encoding: fallback encoding according to
      :func:`encoding_for`
    :return: a :class:`SourceAnalysis`
    """
    assert encoding is not None
    assert generated_regexes is not None

    result = None
    lexer = None
    source_code = None
    source_size = os.path.getsize(source_path)
    if source_size == 0:
        _log.info('%s: is empty', source_path)
        result = pseudo_source_analysis(source_path, group, SourceState.empty)
    elif is_binary_file(source_path):
        _log.info('%s: is binary', source_path)
        result = pseudo_source_analysis(source_path, group, SourceState.binary)
    elif not has_lexer(source_path):
        _log.info('%s: unknown language', source_path)
        result = pseudo_source_analysis(source_path, group, SourceState.unknown)
    elif duplicate_pool is not None:
        duplicate_path = duplicate_pool.duplicate_path(source_path)
        if duplicate_path is not None:
            _log.info('%s: is a duplicate of %s', source_path, duplicate_path)
            result = pseudo_source_analysis(source_path, group, SourceState.duplicate, duplicate_path)
    if result is None:
        if encoding in ('automatic', 'chardet'):
            encoding = encoding_for(source_path, encoding, fallback_encoding)
        try:
            with open(source_path, 'r', encoding=encoding) as source_file:
                source_code = source_file.read()
        except (LookupError, OSError, UnicodeError) as error:
            _log.warning('cannot read %s using encoding %s: %s', source_path, encoding, error)
            result = pseudo_source_analysis(source_path, group, SourceState.error, error)
        if result is None:
            lexer = guess_lexer(source_path, source_code)
            assert lexer is not None
    if (result is None) and (len(generated_regexes) != 0):
        number_line_and_regex = matching_number_line_and_regex(
            pygount.common.lines(source_code), generated_regexes
        )
        if number_line_and_regex is not None:
            number, _, regex = number_line_and_regex
            message = 'line {0} matches {1}'.format(number, regex)
            _log.info('%s: is generated code because %s', source_path, message)
            result = pseudo_source_analysis(source_path, group, SourceState.generated, message)
    if result is None:
        assert lexer is not None
        assert source_code is not None
        language = lexer.name
        if ('xml' in language.lower()) or (language == 'Genshi'):
            dialect = pygount.xmldialect.xml_dialect(source_path, source_code)
            if dialect is not None:
                language = dialect
        _log.info('%s: analyze as %s using encoding %s', source_path, language, encoding)
        mark_to_count_map = {'c': 0, 'd': 0, 'e': 0, 's': 0}
        for line_parts in _line_parts(lexer, source_code):
            mark_to_increment = 'e'
            for mark_to_check in ('d', 's', 'c'):
                if mark_to_check in line_parts:
                    mark_to_increment = mark_to_check
            mark_to_count_map[mark_to_increment] += 1
        result = SourceAnalysis(
            path=source_path,
            language=language,
            group=group,
            code=mark_to_count_map['c'],
            documentation=mark_to_count_map['d'],
            empty=mark_to_count_map['e'],
            string=mark_to_count_map['s'],
            state=SourceState.analyzed.name,
            state_info=None,
        )

    assert result is not None
    return result