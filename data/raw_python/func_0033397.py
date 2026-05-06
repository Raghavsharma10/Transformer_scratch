def add_feature(self, label, value=None):
        """
        label: A VW label (not containing characters from escape_dict.keys(),
            unless 'escape' mode is on)
        value: float giving the weight or magnitude of this feature
        """
        if self.escape:
            label = escape_vw_string(label)
        elif self.validate:
            validate_vw_string(label)
        feature = (label, value)
        self.features.append(feature)