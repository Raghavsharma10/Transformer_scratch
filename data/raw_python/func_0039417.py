def lookup_folder(event, filesystem):
    """Lookup the parent folder in the filesystem content."""
    for dirent in filesystem[event.parent_inode]:
        if dirent.type == 'd' and dirent.allocated:
            return ntpath.join(dirent.path, event.name)