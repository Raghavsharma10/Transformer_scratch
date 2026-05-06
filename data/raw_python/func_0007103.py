def modifie_options(self, field_option, value):
        """Set options in modifications.
        All options will be stored since it should be grouped in the DB."""
        options = dict(self["options"] or {}, **{field_option: value})
        self.modifications["options"] = options