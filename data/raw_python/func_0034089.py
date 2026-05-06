def _part(self, name, func, args, help, **kwargs):
        """Parses arguments of a single command (e.g. 'v').

        If :args: is empty, it assumes that command takes no further arguments.

        :name: Name of the command.
        :func: Arg method to execute.
        :args: Dictionary of CLI arguments pointed at Arg method arguments.
        :help: Commands' help text.
        :kwargs: Additional arguments for :func:.

        """
        while self.argv:
            arg = self.argv.popleft()
            if arg == "-h" or arg == "--help":
                print(help)
                return
            try:
                argname, argarg = args[arg]
                kwargs[argname] = argarg and self.argv.popleft() or True
            except KeyError:
                raise UnrecognizedArgumentError(name, arg)
            except IndexError:
                valids = ["-s", "--sort", "-d", "--done", "-D", "--undone"]
                if arg not in valids:
                    raise NotEnoughArgumentsError(name)
                kwargs[argname] = True
        func(**kwargs)