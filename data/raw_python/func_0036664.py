def extract(dump_files, extractors=ALL_EXTRACTORS):
    """
    Extracts cites from a set of `dump_files`.

    :Parameters:
        dump_files : str | `file`
            A set of files MediaWiki XML dump files
            (expects: pages-meta-history)
        extractors : `list`(`extractor`)
            A list of extractors to apply to the text

    :Returns:
        `iterable` -- a generator of extracted cites

    """
    # Dump processor function
    def process_dump(dump, path):
        for page in dump:
            if page.namespace != 0: continue
            else:
                for cite in extract_cite_history(page, extractors):
                    yield cite

    # Map call
    return mwxml.map(process_dump, dump_files)