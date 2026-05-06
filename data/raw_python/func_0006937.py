def file_attribs(location, mode=None, owner=None, group=None, sudo=False):
    """Updates the mode/owner/group for the remote file at the given
    location."""
    return dir_attribs(location, mode, owner, group, False, sudo)