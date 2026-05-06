def _get_archive_filelist(filename):
    # type: (str) -> List[str]
    """Extract the list of files from a tar or zip archive.

    Args:
        filename: name of the archive

    Returns:
        Sorted list of files in the archive, excluding './'

    Raises:
        ValueError: when the file is neither a zip nor a tar archive
        FileNotFoundError: when the provided file does not exist (for Python 3)
        IOError: when the provided file does not exist (for Python 2)
    """
    names = []  # type: List[str]
    if tarfile.is_tarfile(filename):
        with tarfile.open(filename) as tar_file:
            names = sorted(tar_file.getnames())
    elif zipfile.is_zipfile(filename):
        with zipfile.ZipFile(filename) as zip_file:
            names = sorted(zip_file.namelist())
    else:
        raise ValueError("Can not get filenames from '{!s}'. "
                         "Not a tar or zip file".format(filename))
    if "./" in names:
        names.remove("./")
    return names