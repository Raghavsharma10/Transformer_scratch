def add_subvisual(self, visual):
        """Add a subvisual

        Parameters
        ----------
        visual : instance of Visual
            The visual to add.
        """
        visual.transforms = self.transforms
        visual._prepare_transforms(visual)
        self._subvisuals.append(visual)
        visual.events.update.connect(self._subv_update)
        self.update()