def directory_duplicates(directory, hash_type='md5', **kwargs):
    """
    Find all duplicates in a directory. Will return a list, in that list
    are lists of duplicate files.

    .. code: python

        dups = reusables.directory_duplicates('C:\\Users\\Me\\Pictures')

        print(len(dups))
        # 56
        print(dups)
        # [['C:\\Users\\Me\\Pictures\\IMG_20161127.jpg',
        # 'C:\\Users\\Me\\Pictures\\Phone\\IMG_20161127.jpg'], ...


    :param directory: Directory to search
    :param hash_type: Type of hash to perform
    :param kwargs: Arguments to pass to find_files to narrow file types
    :return: list of lists of dups"""
    size_map, hash_map = defaultdict(list), defaultdict(list)

    for item in find_files(directory, **kwargs):
        file_size = os.path.getsize(item)
        size_map[file_size].append(item)

    for possible_dups in (v for v in size_map.values() if len(v) > 1):
        for each_item in possible_dups:
            item_hash = file_hash(each_item, hash_type=hash_type)
            hash_map[item_hash].append(each_item)

    return [v for v in hash_map.values() if len(v) > 1]