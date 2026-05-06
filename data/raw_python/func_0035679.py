def start(self):
        """
        Start the application, initializing your components.
        """
        current_pedalboard = self.controller(CurrentController).pedalboard
        if current_pedalboard is None:
            self.log('Not exists any current pedalboard.')
            self.log('Use CurrentController to set the current pedalboard')
        else:
            self.log('Load current pedalboard - "{}"', current_pedalboard.name)

        self.mod_host.pedalboard = current_pedalboard

        for component in self.components:
            component.init()
            self.log('Load component - {}', component.__class__.__name__)

        self.log('Components loaded')
        atexit.register(self.stop)