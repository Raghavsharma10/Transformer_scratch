def create_fake_copies(files, destination):
    """
    Create copies of the given list of files in the destination given.

    Creates copies of the actual files to be committed using
    git show :<filename>

    Return a list of destination files.
    """
    dest_files = []
    for filename in files:
        leaf_dest_folder = os.path.join(destination, os.path.dirname(filename))
        if not os.path.exists(leaf_dest_folder):
            os.makedirs(leaf_dest_folder)
        dest_file = os.path.join(destination, filename)
        bash("git show :{filename} > {dest_file}".format(
            filename=filename,
            dest_file=dest_file)
        )
        dest_files.append(os.path.realpath(dest_file))
    return dest_files