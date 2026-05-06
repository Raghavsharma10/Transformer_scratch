def append(self, tr):
        """
        Add a new transform to the end of this chain.

        Parameters
        ----------
        tr : instance of Transform
            The transform to use.
        """
        self.transforms.append(tr)
        tr.changed.connect(self._subtr_changed)
        self._rebuild_shaders()
        self.update()