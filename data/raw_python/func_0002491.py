def call(self, name, options=None, o=None):
        """
        Call another command.

        :param name: The command name
        :type name: str

        :param options: The options
        :type options: list or None

        :param o: The output
        :type o: cleo.outputs.output.Output
        """
        if options is None:
            options = []

        command = self.get_application().find(name)

        options = [('command', command.get_name())] + options

        return command.run(ListInput(options), o)