def get_sudoers_entry(username=None, sudoers_entries=None):
    """ Find the sudoers entry in the sudoers file for the specified user.

    args:
        username (str): username.
        sudoers_entries (list): list of lines from the sudoers file.

    returns:`r
        str: sudoers entry for the specified user.
    """
    for entry in sudoers_entries:
        if entry.startswith(username):
            return entry.replace(username, '').strip()