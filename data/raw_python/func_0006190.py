def get_key(self, command, args):
        """Returns the key a command operates on."""
        spec = COMMANDS.get(command.upper())

        if spec is None:
            raise UnroutableCommand('The command "%r" is unknown to the '
                                    'router and cannot be handled as a '
                                    'result.' % command)

        if 'movablekeys' in spec['flags']:
            raise UnroutableCommand('The keys for "%r" are movable and '
                                    'as such cannot be routed to a single '
                                    'host.')

        keys = extract_keys(args, spec['key_spec'])
        if len(keys) == 1:
            return keys[0]
        elif not keys:
            raise UnroutableCommand(
                'The command "%r" does not operate on a key which means '
                'that no suitable host could be determined.  Consider '
                'using a fanout instead.')

        raise UnroutableCommand(
            'The command "%r" operates on multiple keys (%d passed) which is '
            'something that is not supported.' % (command, len(keys)))