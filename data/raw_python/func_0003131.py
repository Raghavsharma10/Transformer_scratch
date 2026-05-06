def rdf_iterator(graph, lang, add_vtype=False):
    """
    Convert a Graph response into a double iterable, by triples and elements.
    Optionally add element type, and filter triples by language (on literals)
    """
    # Return the header row
    hdr = ('subject', 'predicate', 'object')
    yield hdr if not add_vtype else ((h, 'type') for h in hdr)
    # Now the data rows
    for row in graph:
        if lang and not lang_match_rdf(row, lang):
            continue
        yield ((unicode(c), gtype(c)) for c in row)