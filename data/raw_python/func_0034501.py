def find_match(self):
        """Load the config and find a matching rule.

        returns the results of find_match_command, a dict of
        the command and (in the future) other metadata.
        """

        self.load()
        for yamldoc in self.yamldocs:
            self.logdebug('\nchecking rule """%s"""\n' % yamldoc)

            if not yamldoc:
                continue

            if not self.check_client_ip(yamldoc):
                # Rejected - Client IP does not match
                continue

            if not self.check_keyname(yamldoc):
                # Rejected - keyname does not match
                continue

            rules = yamldoc.get('allow')
            if not isinstance(rules, list):
                rules = [rules]

            for rule in rules:
                rule_type = rule.get('rule_type', 'command')
                if rule_type == 'command':
                    sub = self.find_match_command
                elif rule_type == 'scp':
                    sub = self.find_match_scp
                else:
                    self.log('fatal: no such rule_type "%s"\n' % rule_type)
                    self.raise_and_log_error(ConfigError,
                                             'error parsing config.')

                match = sub(rule)
                if match:
                    return match

        # No matches, time to give up.
        raise CommandRejected('command "%s" denied.' %
                              self.original_command_string)