def list_distribute_contents_simple(input_list, function=lambda x: x):
    # type: (List, Callable[[Any], Any]) -> List
    """Distribute the contents of a list eg. [1, 1, 1, 2, 2, 3] -> [1, 2, 3, 1, 2, 1]. List can contain complex types
    like dictionaries in which case the function can return the appropriate value eg.  lambda x: x[KEY]

    Args:
        input_list (List): List to distribute values
        function (Callable[[Any], Any]): Return value to use for distributing. Defaults to lambda x: x.

    Returns:
        List: Distributed list

    """
    dictionary = dict()
    for obj in input_list:
        dict_of_lists_add(dictionary, function(obj), obj)
    output_list = list()
    i = 0
    done = False
    while not done:
        found = False
        for key in sorted(dictionary):
            if i < len(dictionary[key]):
                output_list.append(dictionary[key][i])
                found = True
        if found:
            i += 1
        else:
            done = True
    return output_list