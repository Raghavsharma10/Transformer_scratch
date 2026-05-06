def seq_seqhash(seq, normalize=True):
    """returns 24-byte Truncated Digest sequence `seq`

    >>> seq_seqhash("")
    'z4PhNX7vuL3xVChQ1m2AB9Yg5AULVxXc'

    >>> seq_seqhash("ACGT")
    'aKF498dAxcJAqme6QYQ7EZ07-fiw8Kw2'

    >>> seq_seqhash("acgt")
    'aKF498dAxcJAqme6QYQ7EZ07-fiw8Kw2'

    >>> seq_seqhash("acgt", normalize=False)
    'eFwawHHdibaZBDcs9kW3gm31h1NNJcQe'

    """

    seq = normalize_sequence(seq) if normalize else seq
    return str(vmc_digest(seq, digest_size=24))