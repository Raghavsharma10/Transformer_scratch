def extract_cite_history(page, extractors):
    """
    Extracts cites from the history of a `page` (`mwxml.Page`).

    :Parameters:
        page : `iterable`(`mwxml.Revision`)
            The page to extract cites from
        extractors : `list`(`extractor`)
            A list of extractors to apply to the text

    :Returns:
        `iterable` -- a generator of extracted cites

    """
    appearances = {} # For tracking the first appearance of an ID
    ids = set() # For holding onto the ids in the last revision.
    for revision in page:
        ids = set(extract_ids(revision.text, extractors))

        # For each ID, check to see if we have seen it before
        for id in ids:
            if id not in appearances:
               appearances[id] = (revision.id, revision.timestamp)

    for id in ids: #For the ids in the last version of the page
        rev_id, timestamp = appearances[id]
        yield (page.id, page.title, rev_id, timestamp, id.type, id.id)