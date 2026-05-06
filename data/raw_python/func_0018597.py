def adjusted_ops(opcodes):
    """
    Iterate through opcodes, turning them into a series of insert and delete
    operations, adjusting indices to account for the size of insertions and
    deletions.

    >>> def sequence_opcodes(old, new): return difflib.SequenceMatcher(a=old, b=new).get_opcodes()
    >>> list(adjusted_ops(sequence_opcodes('abc', 'b')))
    [('delete', 0, 1, 0, 0), ('delete', 1, 2, 1, 1)]
    >>> list(adjusted_ops(sequence_opcodes('b', 'abc')))
    [('insert', 0, 0, 0, 1), ('insert', 2, 2, 2, 3)]
    >>> list(adjusted_ops(sequence_opcodes('axxa', 'aya')))
    [('delete', 1, 3, 1, 1), ('insert', 1, 1, 1, 2)]
    >>> list(adjusted_ops(sequence_opcodes('axa', 'aya')))
    [('delete', 1, 2, 1, 1), ('insert', 1, 1, 1, 2)]
    >>> list(adjusted_ops(sequence_opcodes('ab', 'bc')))
    [('delete', 0, 1, 0, 0), ('insert', 1, 1, 1, 2)]
    >>> list(adjusted_ops(sequence_opcodes('bc', 'ab')))
    [('insert', 0, 0, 0, 1), ('delete', 2, 3, 2, 2)]
    """
    while opcodes:
        op = opcodes.pop(0)
        tag, i1, i2, j1, j2 = op
        shift = 0
        if tag == 'equal':
            continue
        if tag == 'replace':
            # change the single replace op into a delete then insert
            # pay careful attention to the variables here, there's no typo
            opcodes = [
                ('delete', i1, i2, j1, j1),
                ('insert', i2, i2, j1, j2),
            ] + opcodes
            continue
        yield op
        if tag == 'delete':
            shift = -(i2 - i1)
        elif tag == 'insert':
            shift = +(j2 - j1)
        new_opcodes = []
        for tag, i1, i2, j1, j2 in opcodes:
            new_opcodes.append((
                tag,
                i1 + shift,
                i2 + shift,
                j1,
                j2,
            ))
        opcodes = new_opcodes