def parse_rdp_assignment(line):
    """Returns a list of assigned taxa from an RDP classification line
    """
    toks = line.strip().split('\t')
    seq_id = toks.pop(0)
    direction = toks.pop(0)
    if ((len(toks) % 3) != 0):
        raise ValueError(
            "Expected assignments in a repeating series of (rank, name, "
            "confidence), received %s" % toks)
    assignments = []
    # Fancy way to create list of triples using consecutive items from
    # input.  See grouper function in documentation for itertools for
    # more general example.
    itoks = iter(toks)
    for taxon, rank, confidence_str in zip(itoks, itoks, itoks):
        if not taxon:
            continue
        assignments.append((taxon.strip('"'), rank, float(confidence_str)))
    return seq_id, direction, assignments