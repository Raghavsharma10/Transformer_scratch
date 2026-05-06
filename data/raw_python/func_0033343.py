def parse_args_to_action_args(self, argv=None):
        '''
        Parses args and returns an action and the args that were parsed
        '''
        args = self.parse_args(argv)
        action = self.subcommands[args.subcommand][1]
        return action, args