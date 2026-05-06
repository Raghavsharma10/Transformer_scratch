def edit(self, **args):
        """Handles the 'e' command.

        :args: Arguments supplied to the 'e' command.

        """
        if self.model.exists(args["index"]):
            values = dict(zip(
                ['parent', 'name', 'priority', 'comment', 'done'],
                self.model.get(args["index"])
            ))
            kwargs = self.getKwargs(args, values)
            if kwargs:
                self.model.edit(args["index"], **kwargs)