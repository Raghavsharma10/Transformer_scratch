def _process_refs(x, labels):
    """Strips surrounding curly braces and adds modifiers to the
    attributes of Cite elements.  Only references with labels in the 'labels'
    list are processed.  Repeats processing (via decorator) until no more
    broken references are found."""

    # Scan the element list x for Cite elements with known labels
    for i, v in enumerate(x):
        if v['t'] == 'Cite' and len(v['c']) == 2 and \
          _get_label(v['t'], v['c']) in labels:

            # A new reference was found; create some empty attributes for it
            attrs = ['', [], []]

            # Extract the modifiers.  'attrs' is updated in place.  Element
            # deletion could change the index of the Cite being processed.
            if i > 0:
                i = _extract_modifier(x, i, attrs)

            # Attach the attributes
            v['c'].insert(0, attrs)

            # Remove surrounding brackets
            if i > 0 and i < len(x)-1:
                _remove_brackets(x, i)

            # The element list may be changed
            return None  # Forces processing to repeat via _repeat decorator

    return True