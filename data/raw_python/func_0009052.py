def calculate_similarity(container1=None,
                         container2=None,
                         comparison=None,
                         metric=None):

    '''calculate_similarity will calculate similarity of two containers 
    by files content, default will calculate
  
          2.0*len(intersect) / total package1 + total package2

    Parameters
    ==========
    container1: container 1
    container2: container 2 must be defined or
    metric a function to take a total1, total2, and intersect count 
    (we can make this more general if / when more are added)
     valid are currently files.txt or folders.txt
    comparison: the comparison result object for the tree. If provided,
    will skip over function to obtain it.

    '''
    if metric is None:
        metric = information_coefficient

    if comparison == None:
        comparison = compare_containers(container1=container1,
                                        container2=container2)

    return metric(total1=comparison['total1'],
                  total2=comparison['total2'],
                  intersect=comparison["intersect"])