def apply_default_labels(self, other):
        """Applies labels for default meta labels from self onto other.
        
        Parameters
        ----------
        other : Meta
            Meta object to have default labels applied
        
        Returns
        -------
        Meta
        
        """
        other_updated = other.copy()
        other_updated.units_label = self.units_label
        other_updated.name_label = self.name_label
        other_updated.notes_label = self.notes_label
        other_updated.desc_label = self.desc_label
        other_updated.plot_label = self.plot_label
        other_updated.axis_label = self.axis_label
        other_updated.scale_label = self.scale_label
        other_updated.min_label = self.min_label
        other_updated.max_label = self.max_label
        other_updated.fill_label = self.fill_label
        return other