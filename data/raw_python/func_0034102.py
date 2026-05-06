def options(self, glob=False, **args):
        """Handles the 'o' command.

        :glob: Whether to store specified options globally.
        :args: Arguments supplied to the 'o' command (excluding '-g').

        """
        kwargs = {}
        for argname, argarg in args.items():
            if argname == "sort":
                argarg = self._getPattern(argarg)
            if argname not in ["done", "undone"]:
                kwargs[argname] = argarg
        if "done" in args or "undone" in args:
            kwargs["done"] = self._getDone(
                args.get("done"), args.get("undone")
            )

        self.model.setOptions(glob=glob, **kwargs)