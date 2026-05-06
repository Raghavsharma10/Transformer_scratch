def list_actions(cls):
        """Get a list of exposed actions that are callable via the
        ``do_action()`` method."""
        # Make sure these are always at the beginning of the list
        actions = ['start', 'stop', 'restart', 'status']
        # Iterate over the instance attributes checking for actions that
        # have been exposed
        for func_name in dir(cls):
            func = getattr(cls, func_name)
            if (not hasattr(func, '__call__') or
                    not getattr(func, '__daemonocle_exposed__', False)):
                # Not a function or not exposed
                continue
            action = func_name.replace('_', '-')
            if action not in actions:
                actions.append(action)

        return actions