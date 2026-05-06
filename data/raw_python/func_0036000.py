def get_action(self, action):
        """Get a callable action."""
        func_name = action.replace('-', '_')
        if not hasattr(self, func_name):
            # Function doesn't exist
            raise DaemonError(
                'Invalid action "{action}"'.format(action=action))

        func = getattr(self, func_name)
        if (not hasattr(func, '__call__') or
                getattr(func, '__daemonocle_exposed__', False) is not True):
            # Not a function or not exposed
            raise DaemonError(
                'Invalid action "{action}"'.format(action=action))

        return func