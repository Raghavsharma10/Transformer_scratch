def tag_reachable_scripts(cls, scratch):
        """Tag each script with attribute reachable.

        The reachable attribute will be set false for any script that does not
        begin with a hat block. Additionally, any script that begins with a
        'when I receive' block whose event-name doesn't appear in a
        corresponding broadcast block is marked as unreachable.

        """
        if getattr(scratch, 'hairball_prepared', False):  # Only process once
            return

        reachable = set()
        untriggered_events = {}
        # Initial pass to find reachable and potentially reachable scripts
        for script in cls.iter_scripts(scratch):
            if not isinstance(script, kurt.Comment):
                starting_type = cls.script_start_type(script)
                if starting_type == cls.NO_HAT:
                    script.reachable = False
                elif starting_type == cls.HAT_WHEN_I_RECEIVE:
                    # Value will be updated if reachable
                    script.reachable = False
                    message = script[0].args[0].lower()
                    untriggered_events.setdefault(message, set()).add(script)
                else:
                    script.reachable = True
                    reachable.add(script)
        # Expand reachable states based on broadcast events
        while reachable:
            for event in cls.get_broadcast_events(reachable.pop()):
                if event in untriggered_events:
                    for script in untriggered_events.pop(event):
                        script.reachable = True
                        reachable.add(script)
        scratch.hairball_prepared = True