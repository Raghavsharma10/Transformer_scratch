def accept_default_labels(self, other):
        """Applies labels for default meta labels from other onto self.
        
        Parameters
        ----------
        other : Meta
            Meta object to take default labels from
        
        Returns
        -------
        Meta
        
        """

        self.units_label = other.units_label
        self.name_label = other.name_label
        self.notes_label = other.notes_label
        self.desc_label = other.desc_label
        self.plot_label = other.plot_label
        self.axis_label = other.axis_label
        self.scale_label = other.scale_label
        self.min_label = other.min_label
        self.max_label = other.max_label
        self.fill_label = other.fill_label
        return