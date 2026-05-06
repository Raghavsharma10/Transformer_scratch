def get_image_hash(image_path,
                   level=None,level_filter=None,
                   include_files=None,
                   skip_files=None,
                   version=None):

    '''get_image_hash will generate a sha1 hash of an image, depending on a level
    of reproducibility specified by the user. (see function get_levels for descriptions)
    the user can also provide a level_filter manually with level_filter (for custom levels)
    :param level: the level of reproducibility to use, which maps to a set regular
    expression to match particular files/folders in the image. Choices are in notes.
    :param skip_files: an optional list of files to skip
    :param include_files: an optional list of files to keep (only if level not defined)
    :param version: the version to use. If not defined, default is 2.3

    ::notes

    LEVEL DEFINITIONS
    The level definitions come down to including folders/files in the comparison. For files
    that Singularity produces on the fly that might be different (timestamps) but equal content
    (eg for a replication) we hash the content ("assess_content") instead of the file.
    '''    

    # First get a level dictionary, with description and regexp
    if level_filter is not None:
        file_filter = level_filter

    elif level is None:
        file_filter = get_level("RECIPE",
                                version=version,
                                include_files=include_files,
                                skip_files=skip_files)

    else:
        file_filter = get_level(level,version=version,
                                skip_files=skip_files,
                                include_files=include_files)
                
    file_obj, tar = get_image_tar(image_path)
    hasher = hashlib.md5()

    for member in tar:
        member_name = member.name.replace('.','',1)

        # For files, we either assess content, or include the file
        if member.isdir() or member.issym():
            continue
        elif assess_content(member,file_filter):
            content = extract_content(image_path,member.name)
            hasher.update(content)
        elif include_file(member,file_filter):
            buf = member.tobuf()
            hasher.update(buf)

    digest = hasher.hexdigest()

    # Close up / remove files
    try:
        file_obj.close()
    except:
        tar.close()
 
    if os.path.exists(file_obj):
        os.remove(file_obj)

    return digest