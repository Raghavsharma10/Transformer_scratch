def lang_match_xml(row, accepted_languages):
    '''Find if the XML row contains acceptable language data'''
    if not accepted_languages:
        return True
    column_languages = set()
    for elem in row:
        lang = elem[0].attrib.get(XML_LANG, None)
        if lang:
            column_languages.add(lang)
    return (not column_languages) or (column_languages & accepted_languages)