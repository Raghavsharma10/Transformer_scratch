def is_interactive(self):
        """ Determine if the user requested interactive mode.
        """
        # The Python interpreter sets sys.flags correctly, so use them!
        if sys.flags.interactive:
            return True

        # IPython does not set sys.flags when -i is specified, so first
        # check it if it is already imported.
        if '__IPYTHON__' not in dir(six.moves.builtins):
            return False

        # Then we check the application singleton and determine based on
        # a variable it sets.
        try:
            from IPython.config.application import Application as App
            return App.initialized() and App.instance().interact
        except (ImportError, AttributeError):
            return False