def grouparg(self, arg, my_arg=None, parent_cmds=[]):
        """
        Grouper function for chaining subcommands

        Parameters
        ----------
        arg: str
            The current command line argument that is parsed
        my_arg: str
            The name of this subparser. If None, this parser is the main
            parser and has no parent parser
        parent_cmds: list of str
            The available commands of the parent parsers

        Returns
        -------
        str or None
            The grouping key for the given `arg` or None if the key does
            not correspond to this parser or this parser is the main parser
            and does not have seen a subparser yet

        Notes
        -----
        Quite complicated, there is no real need to deal with this function
        """
        if self._subparsers_action is None:
            return None
        commands = self._subparsers_action.choices
        currentarg = self.__currentarg
        # the default return value is the current argument we are in or the
        # name of the subparser itself
        ret = currentarg or my_arg
        if currentarg is not None:
            # if we are already in a sub command, we use the sub parser
            sp_key = commands[currentarg].grouparg(arg, currentarg, chain(
                commands, parent_cmds))
            if sp_key is None and arg in commands:
                # if the subparser did not recognize the command, we use the
                # command the corresponds to this parser or (of this parser
                # is the parent parser) the current subparser
                self.__currentarg = currentarg = arg
                ret = my_arg or currentarg
            elif sp_key not in commands and arg in parent_cmds:
                # otherwise, if the subparser recognizes the commmand but it is
                # not in the known command of this parser, it must be another
                # command of the subparser and this parser can ignore it
                ret = None
            else:
                # otherwise the command belongs to this subparser (if this one
                # is not the subparser) or the current subparser
                ret = my_arg or currentarg
        elif arg in commands:
            # if the argument is a valid subparser, we return this one
            self.__currentarg = arg
            ret = arg
        elif arg in parent_cmds:
            # if the argument is not a valid subparser but in one of our
            # parents, we return None to signalize that we cannot categorize
            # it
            ret = None
        return ret