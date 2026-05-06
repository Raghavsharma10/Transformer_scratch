def get_command(self, ctx, name):
        """Get a callable command object."""
        if name not in self.daemon_class.list_actions():
            return None

        # The context object is a Daemon object
        daemon = ctx.obj

        def subcommand(debug=False):
            """Call a daemonocle action."""
            if daemon.detach and debug:
                daemon.detach = False

            daemon.do_action(name)

        # Override the docstring for the function so that it shows up
        # correctly in the help output
        subcommand.__doc__ = daemon.get_action(name).__doc__

        if name == 'start':
            # Add a --debug option for start
            subcommand = click.option(
                '--debug', is_flag=True,
                help='Do NOT detach and run in the background.'
            )(subcommand)

        # Make it into a click command
        subcommand = click.command(
            name, options_metavar=self.options_metavar)(subcommand)

        return subcommand