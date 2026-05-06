def compare_user(passed_user=None, user_list=None):
    """Check if supplied User instance exists in supplied Users list and, if so, return the differences.

    args:
        passed_user (User): the user instance to check for differences
        user_list (Users): the Users instance containing a list of Users instances

    returns:
        dict: Details of the matching user and a list of differences
    """
    # Check if user exists
    returned = user_list.describe_users(users_filter=dict(name=passed_user.name))
    replace_keys = False
    # User exists, so compare attributes
    comparison_result = dict()
    if passed_user.uid and (not returned[0].uid == passed_user.uid):
        comparison_result['uid_action'] = 'modify'
        comparison_result['current_uid_value'] = returned[0].uid
        comparison_result['replacement_uid_value'] = passed_user.uid
    if passed_user.gid and (not returned[0].gid == passed_user.gid):
        comparison_result['gid_action'] = 'modify'
        comparison_result['current_gid_value'] = returned[0].gid
        comparison_result['replacement_gid_value'] = passed_user.gid
    if passed_user.gecos and (not returned[0].gecos == passed_user.gecos):
        comparison_result['gecos_action'] = 'modify'
        comparison_result['current_gecos_value'] = returned[0].gecos
        comparison_result['replacement_gecos_value'] = passed_user.gecos
    if passed_user.home_dir and (not returned[0].home_dir == passed_user.home_dir):
        comparison_result['home_dir_action'] = 'modify'
        comparison_result['current_home_dir_value'] = returned[0].home_dir
        comparison_result['replacement_home_dir_value'] = passed_user.home_dir
        # (Re)set keys if home dir changed
        replace_keys = True
    if passed_user.shell and (not returned[0].shell == passed_user.shell):
        comparison_result['shell_action'] = 'modify'
        comparison_result['current_shell_value'] = returned[0].shell
        comparison_result['replacement_shell_value'] = passed_user.shell
    if passed_user.sudoers_entry and (not returned[0].sudoers_entry == passed_user.sudoers_entry):
        comparison_result['sudoers_entry_action'] = 'modify'
        comparison_result['current_sudoers_entry'] = returned[0].sudoers_entry
        comparison_result['replacement_sudoers_entry'] = passed_user.sudoers_entry
    # if passed_user.public_keys and (not returned[0].public_keys == passed_user.public_keys):
    existing_keys = returned[0].public_keys
    passed_keys = passed_user.public_keys
    # Check if existing and passed keys exist, and if so, compare
    if all((existing_keys, passed_keys)) and len(existing_keys) == len(passed_user.public_keys):
        # Compare each key, and if any differences, replace
        existing = set(key.raw for key in existing_keys)
        replacement = set(key.raw for key in passed_keys)
        if set.difference(existing, replacement):
            replace_keys = True
    # If not existing keys but keys passed set, then
    elif passed_keys and not existing_keys:
        replace_keys = True
    if replace_keys:
        comparison_result['public_keys_action'] = 'modify'
        comparison_result['current_public_keys_value'] = existing_keys
        comparison_result['replacement_public_keys_value'] = passed_keys
    return dict(state='existing', result=comparison_result, existing_user=returned)