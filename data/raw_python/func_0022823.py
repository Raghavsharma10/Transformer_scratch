def update(self):
        """
        Emit an event to inform listeners that properties of this Node have
        changed. Also request a canvas update.
        """
        self.events.update()
        c = getattr(self, 'canvas', None)
        if c is not None:
            c.update(node=self)