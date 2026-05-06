def kmodels(wordlen: int, k: int, input=None, output=None):
    """Return a circuit taking a wordlen bitvector where only k
    valuations return True. Uses encoding from [1].

    Note that this is equivalent to (~x < k).
    - TODO: Add automated simplification so that the circuits
            are equiv.

    [1]: Chakraborty, Supratik, et al. "From Weighted to Unweighted Model
    Counting." IJCAI. 2015.
    """

    assert 0 <= k < 2**wordlen
    if output is None:
        output = _fresh()

    if input is None:
        input = _fresh()

    input_names = named_indexes(wordlen, input)
    atoms = map(aiger.atom, input_names)

    active = False
    expr = aiger.atom(False)
    for atom, bit in zip(atoms, encode_int(wordlen, k, signed=False)):
        active |= bit
        if not active:  # Skip until first 1.
            continue
        expr = (expr | atom) if bit else (expr & atom)

    return aigbv.AIGBV(
        aig=expr.aig,
        input_map=frozenset([(input, tuple(input_names))]),
        output_map=frozenset([(output, (expr.output,))]),
    )