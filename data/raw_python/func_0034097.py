def add(self, **args):
        """Handles the 'a' command.

        :args: Arguments supplied to the 'a' command.

        """
        kwargs = self.getKwargs(args)
        if kwargs:
            self.model.add(**kwargs)