def remove_subvisual(self, visual):
        """Remove a subvisual

        Parameters
        ----------
        visual : instance of Visual
            The visual to remove.
        """
        visual.events.update.disconnect(self._subv_update)
        self._subvisuals.remove(visual)
        self.update()