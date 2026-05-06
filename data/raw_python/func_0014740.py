def stripIEConditionals(contents, addHtmlIfMissing=True):
    '''
        stripIEConditionals - Strips Internet Explorer conditional statements.

        @param contents <str> - Contents String
        @param addHtmlIfMissing <bool> - Since these normally encompass the "html" element, optionally add it back if missing.
    '''
    allMatches = IE_CONDITIONAL_PATTERN.findall(contents)
    if not allMatches:
        return contents

    for match in allMatches:
        contents = contents.replace(match, '')

    if END_HTML.match(contents) and not START_HTML.match(contents):
        contents = addStartTag(contents, '<html>')

    return contents