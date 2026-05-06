def container_similarity(container1=None,container2=None,image_package1=None,
                         image_package2=None,comparison=None):
    '''container_sim will return a data structure to render an html tree 
    (graph) of the intersection (commonalities) between two images or packages
    :param container1: the first container object
    :param container2: the second container object if either not defined, need
    :param image_package1: a packaged container1 (produced by package)
    :param image_package2: a packaged container2 (produced by package)
    :param comparison: the comparison result object for the tree. If provided,
    will skip over function to obtain it.
    '''
    if comparison == None:
        comparison = compare_containers(container1=container1,
                                        container2=container2,
                                        image_package1=image_package1,
                                        image_package2=image_package2,
                                        by=['files.txt','folders.txt'])
    files = comparison["files.txt"]['intersect']
    folders = comparison['folders.txt']['intersect']
    tree = make_container_tree(folders=folders,
                               files=files)
    return tree