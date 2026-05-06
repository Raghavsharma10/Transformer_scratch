def infer_namespace(ac):
    """Infer the single namespace of the given accession

    This function is convenience wrapper around infer_namespaces().
    Returns:
      * None if no namespaces are inferred
      * The (single) namespace if only one namespace is inferred
      * Raises an exception if more than one namespace is inferred

    >>> infer_namespace("ENST00000530893.6")
    'ensembl'

    >>> infer_namespace("NM_01234.5")
    'refseq'

    >>> infer_namespace("A2BC19")
    'uniprot'

    N.B. The following test is disabled because Python 2 and Python 3
    handle doctest exceptions differently. :-(
    X>>> infer_namespace("P12345")
    Traceback (most recent call last):
    ...
    bioutils.exceptions.BioutilsError: Multiple namespaces possible for P12345

    >>> infer_namespace("BOGUS99") is None
    True

    """

    namespaces = infer_namespaces(ac)
    if not namespaces:
        return None
    if len(namespaces) > 1:
        raise BioutilsError("Multiple namespaces possible for {}".format(ac))
    return namespaces[0]