def get_hashed_filename(name, file, suffix=None):
    """
    Gets a new filename for the provided file of the form
    "oldfilename.hash.ext". If the old filename looks like it already contains a
    hash, it will be replaced (so you don't end up with names like
    "pic.hash.hash.ext")

    """
    basename, hash, ext = split_filename(name)
    file.seek(0)
    new_hash = '.%s' % md5(file.read()).hexdigest()[:12]
    if suffix is not None:
        basename = '%s_%s' % (basename, suffix)
    return '%s%s%s' % (basename, new_hash, ext)