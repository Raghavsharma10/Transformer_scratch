def get_pairtree_prefix(pairtree_store):
    """Returns the prefix given in pairtree_prefix file."""
    prefix_path = os.path.join(pairtree_store, 'pairtree_prefix')
    with open(prefix_path, 'r') as prefixf:
        prefix = prefixf.read().strip()
    return prefix