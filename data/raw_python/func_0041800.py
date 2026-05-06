def pair_tree_creator(meta_id):
    """Splits string into a pairtree path."""
    chunks = []
    for x in range(0, len(meta_id)):
        if x % 2:
            continue
        if (len(meta_id) - 1) == x:
            chunk = meta_id[x]
        else:
            chunk = meta_id[x: x + 2]
        chunks.append(chunk)
    return os.sep + os.sep.join(chunks) + os.sep