def mode(self, channel, add='', remove=''):
        """Add and/or remove modes for a given channel.

        The 'add' and 'remove' arguments may, if specified, be either
        sequences or dictionaries. If a dictionary is specified, the
        corresponding values will be passed as arguments (with expansion
        if necessary - {'b': ['foo','bar']} will result in two bans:
            MODE <channel> +bb foo bar

        (Values for modes which do not take arguments are ignored.)
        """
        if not self.server.in_channel(channel):
            _log.warning("Ignoring request to set modes in channel '%s' "
                         "because we are not in that channel.", channel)
            return

        chanmodes = self._get_chanmodes()
        list_modes, always_arg_modes, set_arg_modes, toggle_modes = chanmodes
        # User privilege levels are not always included in channel modes list
        always_arg_modes |= set(self._get_prefixes().itervalues())

        def _arg_to_list(arg, argument_modes, toggle_modes):
            if not isinstance(arg, dict):
                modes = set(arg)
                invalid_modes = modes - toggle_modes
                if invalid_modes:
                    _log.warning("Ignoring the mode(s) '%s' because they are "
                                 "missing required arguments.",
                                 "".join(invalid_modes))
                return modes & toggle_modes, []

            # Okay, so arg is a dict
            modes_with_args = []
            modes_without_args = set()
            for k,v in arg.iteritems():
                if isinstance(v, str):
                    v = [v]
                if k in argument_modes:
                    for val in v:
                        modes_with_args.append((k,val))
                elif k in toggle_modes:
                    modes_without_args.add(k)
                else:
                    _log.warning("Ignoring request to set channel mode '%s' "
                                 "because it is not a recognized mode.", k)
            return modes_without_args, modes_with_args

        add_modes, add_modes_args = _arg_to_list(
            add, list_modes | always_arg_modes | set_arg_modes, toggle_modes)
        remove_modes, remove_modes_args = _arg_to_list(
            remove, list_modes | always_arg_modes, set_arg_modes | toggle_modes)

        max_arg = self.server.features.get("MODES") or 3

        def _send_modes(op, toggle_modes, arg_modes):
            while toggle_modes or arg_modes:
                modes = "".join(toggle_modes)
                toggle_modes = ""
                now_modes, arg_modes = arg_modes[:max_arg], arg_modes[max_arg:]
                modes += "".join(mode for mode,arg in now_modes)
                modes += "".join(" %s" % arg for mode,arg in now_modes)
                self.send("MODE", channel, "%s%s" % (op, modes))

        _send_modes("+", add_modes, add_modes_args)
        _send_modes("-", remove_modes, remove_modes_args)