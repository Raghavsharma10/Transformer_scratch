def run(self, commands=None, default_command=None, context=None):
        """
            Context: A dict of namespaces as the key, and their objects as the
            value. Used to easily inject code into the shell's runtime env.
        """

        if commands:
            self._commands.update(commands)

            # HACK: Overriding the old shell isn't cool.
            # Should do it by default.
        from alchemist.commands import Shell
        self._commands['shell'] = Shell(context=context)

        if default_command is not None and len(sys.argv) == 1:
            sys.argv.append(default_command)

        try:
            result = self.handle(sys.argv[0], sys.argv[1:])
        except SystemExit as e:
            result = e.code

        sys.exit(result or 0)