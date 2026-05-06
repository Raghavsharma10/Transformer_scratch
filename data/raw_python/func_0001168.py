def build(self, columns):
        """Build the style and fields.

        Parameters
        ----------
        columns : list of str
            Column names.
        """
        self.columns = columns
        default = dict(elements.default("default_"),
                       **_safe_get(self.init_style, "default_", {}))
        self.style = elements.adopt({c: default for c in columns},
                                    self.init_style)

        # Store special keys in _style so that they can be validated.
        self.style["default_"] = default
        self.style["header_"] = self._compose("header_", {"align", "width"})
        self.style["aggregate_"] = self._compose("aggregate_",
                                                 {"align", "width"})
        self.style["separator_"] = _safe_get(self.init_style, "separator_",
                                             elements.default("separator_"))
        lgr.debug("Validating style %r", self.style)
        self.style["width_"] = _safe_get(self.init_style, "width_",
                                         elements.default("width_"))
        elements.validate(self.style)
        self._setup_fields()

        ngaps = len(self.columns) - 1
        self.width_separtor = len(self.style["separator_"]) * ngaps
        lgr.debug("Calculated separator width as %d", self.width_separtor)