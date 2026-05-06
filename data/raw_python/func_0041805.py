def add_to_pairtree(output_path, meta_id):
    """Creates pairtree dir structure within pairtree for new
    element."""
    # create the pair path
    paired_path = pair_tree_creator(meta_id)
    path_append = ''
    # for each directory in the pair path
    for pair_dir in paired_path.split(os.sep):
        # append the pair path together, one directory at a time
        path_append = os.path.join(path_append, pair_dir)
        # append the pair path to the output path
        combined_path = os.path.join(output_path, path_append)
        # if the path doesn't already exist, create it
        if not os.path.exists(combined_path):
            os.mkdir(combined_path)
    return combined_path