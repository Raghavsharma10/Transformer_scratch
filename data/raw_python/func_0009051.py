def compare_lists(list1,list2):
    '''compare lists is the lowest level that drives compare_containers and
    compare_packages. It returns a comparison object (dict) with the unique,
    total, and intersecting things between two lists
    :param list1: the list for container1
    :param list2: the list for container2
    '''
    intersect = list(set(list1).intersection(list2))
    unique1 = list(set(list1).difference(list2))
    unique2 = list(set(list2).difference(list1))

    # Return data structure
    comparison = {"intersect":intersect,
                  "unique1": unique1,
                  "unique2": unique2,
                  "total1": len(list1),
                  "total2": len(list2)}
    return comparison