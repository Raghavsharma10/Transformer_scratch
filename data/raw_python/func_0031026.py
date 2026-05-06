def seq_sha512(seq, normalize=True):
    """returns unicode sequence sha512 hexdigest for sequence `seq`.

    >>> seq_sha512('')
    'cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e'

    >>> seq_sha512('ACGT')
    '68a178f7c740c5c240aa67ba41843b119d3bf9f8b0f0ac36cf701d26672964efbd536d197f51ce634fc70634d1eefe575bec34c83247abc52010f6e2bbdb8253'

    >>> seq_sha512('acgt')
    '68a178f7c740c5c240aa67ba41843b119d3bf9f8b0f0ac36cf701d26672964efbd536d197f51ce634fc70634d1eefe575bec34c83247abc52010f6e2bbdb8253'

    >>> seq_sha512('acgt', normalize=False)
    '785c1ac071dd89b69904372cf645b7826df587534d25c41edb2862e54fb2940d697218f2883d2bf1a11cdaee658c7f7ab945a1cfd08eb26cbce57ee88790250a'

    """

    seq = normalize_sequence(seq) if normalize else seq
    bseq = seq.encode("ascii")
    return hashlib.sha512(bseq).hexdigest()