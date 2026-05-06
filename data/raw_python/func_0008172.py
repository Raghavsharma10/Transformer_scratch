def maybe_split_groups(self, max_groups):
        """
        Check if group lists in add/remove directives should be split and split them if needed
        :param max_groups: Max group list size
        :return: True if at least one command was split, False if none were split
        """
        split_commands = []
        # return True if we split at least once
        maybe_split = False
        valid_step_keys = ['add', 'addRoles', 'remove']
        for command in self.commands:
            # commands are assumed to contain a single key
            step_key, step_args = next(six.iteritems(command))
            if step_key not in valid_step_keys or not isinstance(step_args, dict):
                split_commands.append(command)
                continue
            new_commands = [command]
            while True:
                new_command = {step_key: {}}
                for group_type, groups in six.iteritems(command[step_key]):
                    if len(groups) > max_groups:
                        command[step_key][group_type], new_command[step_key][group_type] = \
                            groups[0:max_groups], groups[max_groups:]
                if new_command[step_key]:
                    new_commands.append(new_command)
                    command = new_command
                    maybe_split = True
                else:
                    break
            split_commands += new_commands
        self.commands = split_commands
        return maybe_split