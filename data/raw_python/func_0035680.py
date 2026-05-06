def stop(self):
        """
        Stop the application, closing your components.
        """
        for component in self.components:
            component.close()
            self.log('Stopping component - {}', component.__class__.__name__)

        for controller in self.controllers.values():
            controller.close()
            self.log('Stopping controller - {}', controller.__class__.__name__)

        atexit.unregister(self.stop)