def find_match_command(self, rule):
        """Return a matching (possibly munged) command, if found in rule."""

        command_string = rule['command']
        command_list = command_string.split()

        self.logdebug('comparing "%s" to "%s"\n' %
                      (command_list, self.original_command_list))
        if rule.get('allow_trailing_args'):
            self.logdebug('allow_trailing_args is true - comparing initial '
                          'list.\n')
            # Verify the initial arguments are all the same
            if (self.original_command_list[:len(command_list)] ==
                    command_list):
                self.logdebug('initial list is same\n')
                return {'command': self.original_command_list}
            else:
                self.logdebug('initial list is not same\n')

        elif rule.get('pcre_match'):
            if re.search(command_string, self.original_command_string):
                return {'command': self.original_command_list}

        elif command_list == self.original_command_list:
            return {'command': command_list}