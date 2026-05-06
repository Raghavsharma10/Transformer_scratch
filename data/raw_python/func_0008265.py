def act(self):
        """
        Carries out the action associated with the Unfreeze button
        """
        g = get_root(self).globals
        g.ipars.unfreeze()
        g.rpars.unfreeze()
        g.observe.load.enable()
        self.disable()