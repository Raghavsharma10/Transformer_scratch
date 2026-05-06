def prepend(self, tr):
        """
        Add a new transform to the beginning of this chain.

        Parameters
        ----------
        tr : instance of Transform
            The transform to use.
        """
        self.transforms.insert(0, tr)
        tr.changed.connect(self._subtr_changed)
        self._rebuild_shaders()
        self.update()