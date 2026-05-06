def gen_subcommand_help(self):
        '''
        Generates s
        '''
        commands = sorted(self.subcommands.items(), key=lambda i: i[0])
        return '\n'.join(
            '%s %s' % (
                subcommand.ljust(15),
                textwrap.shorten(description, width=61),
            )
            for subcommand, (description, action, opts) in commands
        )