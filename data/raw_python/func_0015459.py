def partition_t(T: RDFGraph, nparts: int) -> Iterator[Tuple[RDFGraph, ...]]:
    """
    Partition T into all possible partitions of T of size nparts
    :param T: Set of RDF triples to be partitioned
    :param nparts: number of partitions (e.g. 2 means return all possible 2 set partitions
    :return: Iterator that returns partitions

    We don't actually partition the triples directly -- instead, we partition a set of integers that
    reference elements in the (ordered) set and return those
    """
    def partition_map(partition: List[List[int]]) -> Tuple[RDFGraph, ...]:
        rval: List[RDFGraph, ...] = []
        for part in partition:
            if len(part) == 1 and part[0] >= t_list_len:
                rval.append(RDFGraph())
            else:
                rval.append(RDFGraph([t_list[e] for e in part if e < t_list_len]))
        return tuple(rval)

    t_list = sorted(list(T))      # Sorted not strictly necessary, but aids testing
    t_list_len = len(t_list)
    return map(lambda partition: partition_map(partition), filtered_integer_partition(t_list_len, nparts))