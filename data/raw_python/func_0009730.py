def main():
    """Filters the document AST."""

    # pylint: disable=global-statement
    global PANDOCVERSION
    global AttrTable

    # Get the output format and document
    fmt = args.fmt
    doc = json.loads(STDIN.read())

    # Initialize pandocxnos
    # pylint: disable=too-many-function-args
    PANDOCVERSION = pandocxnos.init(args.pandocversion, doc)

    # Element primitives
    AttrTable = elt('Table', 6)

    # Chop up the doc
    meta = doc['meta'] if PANDOCVERSION >= '1.18' else doc[0]['unMeta']
    blocks = doc['blocks'] if PANDOCVERSION >= '1.18' else doc[1:]

    # Process the metadata variables
    process(meta)

    # First pass
    detach_attrs_table = detach_attrs_factory(Table)
    insert_secnos = insert_secnos_factory(Table)
    delete_secnos = delete_secnos_factory(Table)
    altered = functools.reduce(lambda x, action: walk(x, action, fmt, meta),
                               [attach_attrs_table, insert_secnos,
                                process_tables, delete_secnos,
                                detach_attrs_table], blocks)

    # Second pass
    process_refs = process_refs_factory(references.keys())
    replace_refs = replace_refs_factory(references,
                                        use_cleveref_default, False,
                                        plusname if not capitalize else
                                        [name.title() for name in plusname],
                                        starname, 'table')
    altered = functools.reduce(lambda x, action: walk(x, action, fmt, meta),
                               [repair_refs, process_refs, replace_refs],
                               altered)

    # Insert supporting TeX
    if fmt in ['latex']:

        rawblocks = []

        if has_unnumbered_tables:
            rawblocks += [RawBlock('tex', TEX0),
                          RawBlock('tex', TEX1),
                          RawBlock('tex', TEX2)]

        if captionname != 'Table':
            rawblocks += [RawBlock('tex', TEX3 % captionname)]

        insert_rawblocks = insert_rawblocks_factory(rawblocks)

        altered = functools.reduce(lambda x, action: walk(x, action, fmt, meta),
                                   [insert_rawblocks], altered)

    # Update the doc
    if PANDOCVERSION >= '1.18':
        doc['blocks'] = altered
    else:
        doc = doc[:1] + altered

    # Dump the results
    json.dump(doc, STDOUT)

    # Flush stdout
    STDOUT.flush()