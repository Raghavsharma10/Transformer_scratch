def attr_sep(self, new_sep: str) -> None:
        """Set the new value for the attribute separator.
        
        When the new value is assigned a new tree is generated.
        """
        self._attr_sep = new_sep
        self._filters_tree = self._generate_filters_tree()