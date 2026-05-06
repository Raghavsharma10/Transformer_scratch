def get_pair_path(meta_id):
    """Determines the pair path for the digital object meta-id."""
    pair_tree = pair_tree_creator(meta_id)
    pair_path = os.path.join(pair_tree, meta_id)
    return pair_path