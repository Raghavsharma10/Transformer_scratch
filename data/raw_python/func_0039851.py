def command(self, *args, **kwargs):
        """
        Attach a command to the current :class:`CLI` object.

        The function should accept an instance of an
        :class:`argparse.ArgumentParser` and use it to define extra
        arguments and options. These options will only affect the specified
        command.

        """

        def wrapper(func):
            self.add_command(func, *args, **kwargs)
            return func

        return wrapper