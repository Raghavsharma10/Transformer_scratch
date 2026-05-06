def relative_playbook(playbook):
    """ Returns a tuple (controlled, playbook).

    - controlled is a boolean indicating whether or not we think that the
      playbook being run was checked in to our ansible git repo.
    - playbook is the relative file path of the playbook.
    """
    if playbook.startswith(fs_prefix):
        return True, playbook[len(fs_prefix):]
    else:
        return False, playbook.split('/')[-1]