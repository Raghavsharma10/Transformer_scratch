def container_tree(container=None,image_package=None):
    '''tree will render an html tree (graph) of a container
    '''

    guts = get_container_contents(container=container,
                                  image_package=image_package,
                                  split_delim="\n")

    # Make the tree and return it
    tree = make_container_tree(folders = guts["folders.txt"],
                               files = guts['files.txt'])
    return tree