def seq_vmc_identifier(seq, normalize=True):
    """returns VMC identifier (record) for sequence `seq`

    See https://github.com/ga4gh/vmc

    >>> seq_vmc_identifier("") == {'namespace': 'VMC', 'accession': 'GS_z4PhNX7vuL3xVChQ1m2AB9Yg5AULVxXc'}
    True

    >>> seq_vmc_identifier("ACGT") == {'namespace': 'VMC', 'accession': 'GS_aKF498dAxcJAqme6QYQ7EZ07-fiw8Kw2'}
    True

    >>> seq_vmc_identifier("acgt") == {'namespace': 'VMC', 'accession': 'GS_aKF498dAxcJAqme6QYQ7EZ07-fiw8Kw2'}
    True

    >>> seq_vmc_identifier("acgt", normalize=False) == {'namespace': 'VMC', 'accession': 'GS_eFwawHHdibaZBDcs9kW3gm31h1NNJcQe'}
    True

    """
    seq = normalize_sequence(seq) if normalize else seq
    return {"namespace": "VMC", "accession": "GS_" + str(vmc_digest(seq))}