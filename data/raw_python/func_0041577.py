def listIDs(basedir):
    """Lists digital object identifiers of Pairtree directory structure.

    Walks a Pairtree directory structure to get IDs. Prepends prefix
    found in pairtree_prefix file. Outputs to standard output.
    """
    prefix = ''
    # check for pairtree_prefix file
    prefixfile = os.path.join(basedir, 'pairtree_prefix')
    if os.path.isfile(prefixfile):
        rff = open(prefixfile, 'r')
        prefix = rff.readline().strip()
        rff.close()
    # check for pairtree_root dir
    root = os.path.join(basedir, 'pairtree_root')
    if os.path.isdir(root):
        objects = pairtree.findObjects(root)
        for obj in objects:
            doi = os.path.split(obj)[1]
            # print with prefix and original chars in place
            print(prefix + pairtree.deSanitizeString(doi))
    else:
        print('pairtree_root directory not found')