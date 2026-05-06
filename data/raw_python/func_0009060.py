def get_image_hashes(image_path, version=None, levels=None):
    '''get_image_hashes returns the hash for an image across all levels. This is the quickest,
    easiest way to define a container's reproducibility on each level.
    '''
    if levels is None:
        levels = get_levels(version=version)
    hashes = dict()
    for level_name,level_filter in levels.items():
        hashes[level_name] = get_image_hash(image_path,
                                            level_filter=level_filter)
    return hashes