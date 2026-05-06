def _parse_mode(client, command, actor, args):
    """Parse a mode changes, update states, and dispatch MODE events."""
    chantypes = client.server.features.get("CHANTYPES", "#")
    channel, _, args = args.partition(" ")
    args = args.lstrip(":")

    if channel[0] not in chantypes:
        # Personal modes
        for modes in args.split():
            op, modes = modes[0], modes[1:]
            for mode in modes:
                if op == "+":
                    client.user.modes.add(mode)
                else:
                    client.user.modes.discard(mode)
                client.dispatch_event("MODE", actor, client.user, op, mode, None)
        return

    # channel-specific modes
    chan = client.server.get_channel(channel)

    user_modes = set(client._get_prefixes().itervalues())

    chanmodes = client._get_chanmodes()
    list_modes, always_arg_modes, set_arg_modes, toggle_modes = chanmodes
    argument_modes = list_modes | always_arg_modes | set_arg_modes

    tokens = args.split()
    while tokens:
        modes, tokens = tokens[0], tokens[1:]
        op, modes = modes[0], modes[1:]

        for mode in modes:
            argument = None
            if mode in (user_modes | argument_modes):
                argument, tokens = tokens[0], tokens[1:]

            if mode in user_modes:
                user = client.server.get_channel(channel).members[argument]
                if op == "+":
                    user.modes.add(mode)
                else:
                    user.modes.discard(mode)

            if op == "+":
                if mode in (always_arg_modes | set_arg_modes):
                    chan.modes[mode] = argument
                elif mode in toggle_modes:
                    chan.modes[mode] = True
            else:
                if mode in (always_arg_modes | set_arg_modes | toggle_modes):
                    if mode in chan.modes:
                        del chan.modes[mode]

            # list-type modes (bans+exceptions, invite masks) aren't stored,
            # but do generate MODE events.
            client.dispatch_event("MODE", actor, chan, op, mode, argument)