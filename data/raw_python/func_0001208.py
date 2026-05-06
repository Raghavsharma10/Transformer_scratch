def create_plan(existing_users=None, proposed_users=None, purge_undefined=None, protected_users=None,
                allow_non_unique_id=None, manage_home=True, manage_keys=True):
    """Determine what changes are required.

    args:
        existing_users (Users): List of discovered users
        proposed_users (Users): List of proposed users
        purge_undefined (bool): Remove discovered users that have not been defined in proposed users list
        protected_users (list): List of users' names that should not be evaluated as part of the plan creation process
        allow_non_unique_id (bool): Allow more than one user to have the same uid
        manage_home (bool): Create/remove users' home directories
        manage_keys (bool): Add/update/remove users' keys (manage_home must also be true)

    returns:
       list: Differences between discovered and proposed users with a
             list of operations that will achieve the desired state.
    """

    plan = list()
    proposed_usernames = list()

    if not purge_undefined:
        purge_undefined = constants.PURGE_UNDEFINED
    if not protected_users:
        protected_users = constants.PROTECTED_USERS
    if not allow_non_unique_id:
        allow_non_unique_id = constants.ALLOW_NON_UNIQUE_ID

    # Create list of modifications to make based on proposed users compared to existing users
    for proposed_user in proposed_users:
        proposed_usernames.append(proposed_user.name)
        user_matching_name = existing_users.describe_users(users_filter=dict(name=proposed_user.name))
        user_matching_id = get_user_by_uid(uid=proposed_user.uid, users=existing_users)
        # If user does not exist
        if not allow_non_unique_id and user_matching_id and not user_matching_name:
            plan.append(
                dict(action='fail', error='uid_clash', proposed_user=proposed_user, state='existing', result=None))
        elif not user_matching_name:
            plan.append(
                dict(action='add', proposed_user=proposed_user, state='missing', result=None, manage_home=manage_home,
                     manage_keys=manage_keys))
        # If they do, then compare
        else:
            user_comparison = compare_user(passed_user=proposed_user, user_list=existing_users)
            if user_comparison.get('result'):
                plan.append(
                    dict(action='update', proposed_user=proposed_user, state='existing',
                         user_comparison=user_comparison, manage_home=manage_home, manage_keys=manage_keys))
    # Application of the proposed user list will not result in deletion of users that need to be removed
    # If 'PURGE_UNDEFINED' then look for existing users that are not defined in proposed usernames and mark for removal
    if purge_undefined:
        for existing_user in existing_users:
            if existing_user.name not in proposed_usernames:
                if existing_user.name not in protected_users:
                    plan.append(
                        dict(action='delete', username=existing_user.name, state='existing', manage_home=manage_home,
                             manage_keys=manage_keys))
    return plan