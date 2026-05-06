def integer_partition(size: int, nparts: int) -> Iterator[List[List[int]]]:
    """ Partition a list of integers into a list of partitions """
    for part in algorithm_u(range(size), nparts):
        yield part