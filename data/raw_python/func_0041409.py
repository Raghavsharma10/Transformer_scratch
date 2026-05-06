def find_manifest_changes(new_file_state, old_file_state):
    """ Find what has changed between two sets of files """
    prev_state_dict = copy.deepcopy(old_file_state)
    changed_files = {}

    # Find files which are new on the server
    for itm in new_file_state:
        if itm['path'] in prev_state_dict:
            d_itm = prev_state_dict.pop(itm['path'])

            # If the file has been modified
            if itm['last_mod'] != d_itm['last_mod']:
                n_itm = itm.copy()
                n_itm['status'] = 'changed'
                changed_files[itm['path']] = n_itm
            else:
                pass # The file has not changed

        else:
            n_itm = itm.copy()
            n_itm['status'] = 'new'
            changed_files[itm['path']] = n_itm

    # any files remaining in the old file state have been deleted locally
    for itm in prev_state_dict.itervalues():
        n_itm = itm.copy()
        n_itm['status'] = 'deleted'
        changed_files[itm['path']] = n_itm

    return changed_files