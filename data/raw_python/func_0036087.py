def multi_way_partitioning(items, bin_count): #TODO rename bin_count -> bins
    '''
    Greedily divide weighted items equally across bins.

    This approximately solves a multi-way partition problem, minimising the
    difference between the largest and smallest sum of weights in a bin.

    Parameters
    ----------
    items : ~typing.Iterable[~typing.Tuple[~typing.Any, float]]
        Weighted items as ``(item, weight)`` tuples.
    bin_count : int
        Number of bins.

    Returns
    -------
    bins : ~collections_extended.frozenbag[~collections_extended.frozenbag[~typing.Any]]
        Item bins as a bag of item bags.

    Notes
    ----------
    - `A greedy solution <http://stackoverflow.com/a/6855546/1031434>`_
    - `Problem definition and solutions <http://ijcai.org/Proceedings/09/Papers/096.pdf>`_
    '''
    bins = [_Bin() for _ in range(bin_count)]
    for item, weight in sorted(items, key=lambda x: x[1], reverse=True):
        bin_ = min(bins, key=lambda bin_: bin_.weights_sum) 
        bin_.add(item, weight)
    return frozenbag(frozenbag(bin_.items) for bin_ in bins)