def get_value(self, field, quick):
        # type: (Field, bool) -> Any
        """ Ask user the question represented by this instance.

        Args:
            field (Field):
                The field we're asking the user to provide the value for.
            quick (bool):
                Enable quick mode. In quick mode, the form will reduce the
                number of question asked by using defaults wherever possible.
                This can greatly reduce the number of interactions required on
                the user part, but will obviously limit the user choices. This
                should probably be enabled only by a specific user action
                (like passing a ``--quick`` flag etc.).

        Returns:
            The user response converted to a python type using the
            :py:attr:`cliform.core.Field.type` converter.
        """
        if callable(field.default):
            default = field.default(self)
        else:
            default = field.default

        if quick and default is not None:
            return default

        shell.cprint('<90>{}', field.help)

        while True:
            try:
                answer = click.prompt(field.pretty_prompt, default=default)
                return field.type(answer)
            except ValueError:
                shell.cprint("<31>Unsupported value")