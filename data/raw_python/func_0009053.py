def package_node(root=None, name=None):
    '''package node aims to package a (present working node) for a user into
    a container. This assumes that the node is a single partition.
  
    :param root: the root of the node to package, default is /
    :param name: the name for the image. If not specified, will use machine's

    psutil.disk_partitions()

    '''

    if name is None:
        name = platform.node()

    if root is None:
        root = "/"

    tmpdir = tempfile.mkdtemp()
    image = "%s/%s.tgz" %(tmpdir,name)

    print("Preparing to package root %s into %s" %(root,name))
    cmd = ["tar","--one-file-system","-czvSf", image, root,"--exclude",image]
    output = run_command(cmd)
    return image