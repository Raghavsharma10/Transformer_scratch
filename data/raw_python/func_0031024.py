def seq_md5(seq, normalize=True):
    """returns unicode md5 as hex digest for sequence `seq`.

    >>> seq_md5('')
    'd41d8cd98f00b204e9800998ecf8427e'

    >>> seq_md5('ACGT')
    'f1f8f4bf413b16ad135722aa4591043e'

    >>> seq_md5('ACGT*')
    'f1f8f4bf413b16ad135722aa4591043e'

    >>> seq_md5(' A C G T ')
    'f1f8f4bf413b16ad135722aa4591043e'

    >>> seq_md5('acgt')
    'f1f8f4bf413b16ad135722aa4591043e'

    >>> seq_md5('acgt', normalize=False)
    'db516c3913e179338b162b2476d1c23f'

    """
    seq = normalize_sequence(seq) if normalize else seq
    bseq = seq.encode("ascii")
    return hashlib.md5(bseq).hexdigest()