def default_decode(events, mode='full'):
    """Decode a XigtCorpus element."""
    event, elem = next(events)
    root = elem  # store root for later instantiation
    while (event, elem.tag) not in [('start', 'igt'), ('end', 'xigt-corpus')]:
        event, elem = next(events)
    igts = None
    if event == 'start' and elem.tag == 'igt':
        igts = (
            decode_igt(e)
            for e in iter_elements(
                'igt', events, root, break_on=[('end', 'xigt-corpus')]
            )
        )
    xc = decode_xigtcorpus(root, igts=igts, mode=mode)
    return xc