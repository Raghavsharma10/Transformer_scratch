def json_iterator(hdr, rowlist, lang, add_vtype=False):
    """
    Convert a JSON response into a double iterable, by rows and columns
    Optionally add element type, and filter triples by language (on literals)
    """
    # Return the header row
    yield hdr if not add_vtype else ((h, 'type') for h in hdr)
    # Now the data rows
    for row in rowlist:
        if lang and not lang_match_json(row, hdr, lang):
            continue
        yield ((row[c]['value'], jtype(row[c])) if c in row else ('', '')
               for c in hdr)