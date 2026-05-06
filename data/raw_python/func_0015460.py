def partition_2(T: RDFGraph) -> List[Tuple[RDFGraph, RDFGraph]]:
    """
    Partition T into all possible combinations of two subsets
    :param T: RDF Graph to partition
    :return:
    """
    for p in partition_t(T, 2):
        yield p