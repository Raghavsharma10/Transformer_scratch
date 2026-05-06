def list_distribute_contents(input_list, function=lambda x: x):
    # type: (List, Callable[[Any], Any]) -> List
    """Distribute the contents of a list eg. [1, 1, 1, 2, 2, 3] -> [1, 2, 1, 2, 1, 3]. List can contain complex types
    like dictionaries in which case the function can return the appropriate value eg.  lambda x: x[KEY]

    Args:
        input_list (List): List to distribute values
        function (Callable[[Any], Any]): Return value to use for distributing. Defaults to lambda x: x.

    Returns:
        List: Distributed list

    """

    def riffle_shuffle(piles_list):
        def grouper(n, iterable, fillvalue=None):
            args = [iter(iterable)] * n
            return zip_longest(fillvalue=fillvalue, *args)

        if not piles_list:
            return []
        piles_list.sort(key=len, reverse=True)
        width = len(piles_list[0])
        pile_iters_list = [iter(pile) for pile in piles_list]
        pile_sizes_list = [[pile_position] * len(pile) for pile_position, pile in enumerate(piles_list)]
        grouped_rows = grouper(width, itertools.chain.from_iterable(pile_sizes_list))
        grouped_columns = zip_longest(*grouped_rows)
        shuffled_pile = [next(pile_iters_list[position]) for position in itertools.chain.from_iterable(grouped_columns)
                         if position is not None]
        return shuffled_pile

    dictionary = dict()
    for obj in input_list:
        dict_of_lists_add(dictionary, function(obj), obj)
    intermediate_list = list()
    for key in sorted(dictionary):
        intermediate_list.append(dictionary[key])
    return riffle_shuffle(intermediate_list)