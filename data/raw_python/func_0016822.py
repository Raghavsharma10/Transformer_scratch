def reinitialize_command(self, command, reinit_subcommands):
    """
    Monkeypatch distutils.Distribution.reinitialize_command() to match behavior
    of Distribution.get_command_obj()
    This fixes a problem where 'pip install -e' does not reinitialise options
    using the setup(options={...}) variable for the build_ext command.
    This also effects other option sourcs such as setup.cfg.
    """
    cmd_obj = _DISTUTILS_REINIT(self, command, reinit_subcommands)

    options = self.command_options.get(command)

    if options:
        self._set_command_options(cmd_obj, options)

    return cmd_obj