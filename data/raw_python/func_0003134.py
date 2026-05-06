def xml_iterator(columns, rowlist, lang, add_vtype=False):
    """
    Convert an XML response into a double iterable, by rows and columns
    Options are: filter triples by language (on literals), add element type
    """
    # Return the header row
    yield columns if not add_vtype else ((h, 'type') for h in columns)
    # Now the data rows
    for row in rowlist:
        if not lang_match_xml(row, lang):
            continue
        rowdata = {nam: val for nam, val in xml_row(row, lang)}
        yield (rowdata.get(field, ('', '')) for field in columns)