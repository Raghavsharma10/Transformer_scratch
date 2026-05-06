def create_paired_dir(output_dir, meta_id, static=False, needwebdir=True):
    """Creates the meta or static dirs.

    Adds an "even" or "odd" subdirectory to the static path
    based on the meta-id.
    """
    # get the absolute root path
    root_path = os.path.abspath(output_dir)
    # if it's a static directory, add even and odd
    if static:
        # determine whether meta-id is odd or even
        if meta_id[-1].isdigit():
            last_character = int(meta_id[-1])
        else:
            last_character = ord(meta_id[-1])
        if last_character % 2 == 0:
            num_dir = 'even'
        else:
            num_dir = 'odd'
        # add odd or even to the path, based on the meta-id
        output_path = os.path.join(root_path, num_dir)
    # if it's a meta directory, output as normal
    else:
        output_path = root_path
    # if it doesn't already exist, create the output path (includes even/odd)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    # add the pairtree to the output path
    path_name = add_to_pairtree(output_path, meta_id)
    # add the meta-id directory to the end of the pairpath
    meta_dir = os.path.join(path_name, meta_id)
    os.mkdir(meta_dir)
    # if we are creating static output
    if static and needwebdir:
        # add the web path to the output directory
        os.mkdir(os.path.join(meta_dir, 'web'))
        static_dir = os.path.join(meta_dir, 'web')
        return static_dir
    # else we are creating meta output or don't need web directory
    else:
        return meta_dir