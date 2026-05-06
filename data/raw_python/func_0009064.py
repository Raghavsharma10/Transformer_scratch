def container_difference(container=None,container_subtract=None,image_package=None,
                         image_package_subtract=None,comparison=None):
    '''container_difference will return a data structure to render an html 
    tree (graph) of the differences between two images or packages. The second
    container is subtracted from the first
    :param container: the primary container object (to subtract from)
    :param container_subtract: the second container object to remove
    :param image_package: a zipped package for image 1, created with package
    :param image_package_subtract: a zipped package for subtraction image, created with package
    :param comparison: the comparison result object for the tree. If provided,
    will skip over function to obtain it.
    '''
    if comparison == None:
        comparison = compare_containers(container1=container,
                                        container2=container_subtract,
                                        image_package1=image_package,
                                        image_package2=image_package_subtract,
                                        by=['files.txt','folders.txt'])

    files = comparison["files.txt"]['unique1']
    folders = comparison['folders.txt']['unique1']
    tree = make_container_tree(folders=folders,
                               files=files)
    return tree