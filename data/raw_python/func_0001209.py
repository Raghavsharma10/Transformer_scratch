def execute_plan(plan=None):
    """Create, Modify or Delete, depending on plan item."""
    execution_result = list()
    for task in plan:
        action = task['action']
        if action == 'delete':
            command = generate_delete_user_command(username=task.get('username'), manage_home=task['manage_home'])
            command_output = execute_command(command)
            execution_result.append(dict(task=task, command_output=command_output))
            remove_sudoers_entry(username=task.get('username'))
        elif action == 'add':
            command = generate_add_user_command(proposed_user=task.get('proposed_user'), manage_home=task['manage_home'])
            command_output = execute_command(command)
            if task['proposed_user'].public_keys and task['manage_home'] and task['manage_keys']:
                write_authorized_keys(task['proposed_user'])
            if task['proposed_user'].sudoers_entry:
                write_sudoers_entry(username=task['proposed_user'].name,
                                    sudoers_entry=task['proposed_user'].sudoers_entry)
            execution_result.append(dict(task=task, command_output=command_output))
        elif action == 'update':
            result = task['user_comparison'].get('result')
            # Don't modify user if only keys have changed
            action_count = 0
            for k, _ in iteritems(result):
                if '_action' in k:
                    action_count += 1
            command_output = None
            if task['manage_home'] and task['manage_keys'] and action_count == 1 and 'public_keys_action' in result:
                write_authorized_keys(task['proposed_user'])
            elif action_count == 1 and 'sudoers_entry_action' in result:
                write_sudoers_entry(username=task['proposed_user'].name,
                                    sudoers_entry=task['user_comparison']['result']['replacement_sudoers_entry'])
            else:
                command = generate_modify_user_command(task=task)
                command_output = execute_command(command)
                if task['manage_home'] and task['manage_keys'] and result.get('public_keys_action'):
                    write_authorized_keys(task['proposed_user'])
                if result.get('sudoers_entry_action'):
                    write_sudoers_entry(username=task['proposed_user'].name,
                                        sudoers_entry=task['user_comparison']['result']['replacement_sudoers_entry'])
            execution_result.append(dict(task=task, command_output=command_output))