def register_event(self, event):
        """Registers a new command line interface event hook as command"""

        self.log('Registering event hook:', event.cmd, event.thing,
                 pretty=True, lvl=verbose)
        self.hooks[event.cmd] = event.thing