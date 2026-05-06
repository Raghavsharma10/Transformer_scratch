def _repair_refs(x):
    """Performs the repair on the element list 'x'."""

    if _PANDOCVERSION is None:
        raise RuntimeError('Module uninitialized.  Please call init().')

    # Scan the element list x
    for i in range(len(x)-1):

        # Check for broken references
        if _is_broken_ref(x[i]['t'], x[i]['c'] if 'c' in x[i] else [],
                          x[i+1]['t'], x[i+1]['c'] if 'c' in x[i+1] else []):

            # Get the reference string
            n = 0 if _PANDOCVERSION < '1.16' else 1
            s = x[i]['c'][n][0]['c'] + x[i+1]['c']

            # Chop it into pieces.  Note that the prefix and suffix may be
            # parts of other broken references.
            prefix, label, suffix = _REF.match(s).groups()

            # Insert the suffix, label and prefix back into x.  Do it in this
            # order so that the indexing works.
            if suffix:
                x.insert(i+2, Str(suffix))
            x[i+1] = Cite(
                [{"citationId":label,
                  "citationPrefix":[],
                  "citationSuffix":[],
                  "citationNoteNum":0,
                  "citationMode":{"t":"AuthorInText", "c":[]},
                  "citationHash":0}],
                [Str('@' + label)])
            if prefix:
                if i > 0 and x[i-1]['t'] == 'Str':
                    x[i-1]['c'] = x[i-1]['c'] + prefix
                    del x[i]
                else:
                    x[i] = Str(prefix)
            else:
                del x[i]

            return None  # Forces processing to repeat

    return True