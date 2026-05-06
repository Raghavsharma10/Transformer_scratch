def infer_namespaces(ac):
    """infer possible namespaces of given accession based on syntax
    Always returns a list, possibly empty

    >>> infer_namespaces("ENST00000530893.6")
    ['ensembl']
    >>> infer_namespaces("ENST00000530893")
    ['ensembl']
    >>> infer_namespaces("ENSQ00000530893")
    []
    >>> infer_namespaces("NM_01234")
    ['refseq']
    >>> infer_namespaces("NM_01234.5")
    ['refseq']
    >>> infer_namespaces("NQ_01234.5")
    []
    >>> infer_namespaces("A2BC19")
    ['uniprot']
    >>> sorted(infer_namespaces("P12345"))
    ['insdc', 'uniprot']
    >>> infer_namespaces("A0A022YWF9")
    ['uniprot']


    """
    return [v for k, v in ac_namespace_regexps.items() if k.match(ac)]