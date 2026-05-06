def get_content_hashes(image_path,
                       level=None,
                       regexp=None,
                       include_files=None,
                       tag_root=True,
                       level_filter=None,
                       skip_files=None,
                       version=None,
                       include_sizes=True):

    '''get_content_hashes is like get_image_hash, but it returns a complete dictionary 
    of file names (keys) and their respective hashes (values). This function is intended
    for more research purposes and was used to generate the levels in the first place.
    If include_sizes is True, we include a second data structure with sizes
    '''    

    if level_filter is not None:
        file_filter = level_filter

    elif level is None:
        file_filter = get_level("REPLICATE",version=version,
                                skip_files=skip_files,
                                include_files=include_files)

    else:
        file_filter = get_level(level,version=version,
                                skip_files=skip_files,
                                include_files=include_files)

    file_obj,tar = get_image_tar(image_path)

    results = extract_guts(image_path=image_path,
                           tar=tar,
                           file_filter=file_filter,
                           tag_root=tag_root,
                           include_sizes=include_sizes)

    delete_image_tar(file_obj, tar)
    return results