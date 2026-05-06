def render(self, row, style=None, adopt=True):
        """Render fields with values from `row`.

        Parameters
        ----------
        row : dict
            A normalized row.
        style : dict, optional
            A style that follows the schema defined in pyout.elements.  If
            None, `self.style` is used.
        adopt : bool, optional
            Merge `self.style` and `style`, using the latter's keys
            when there are conflicts.  If False, treat `style` as a
            standalone style.

        Returns
        -------
        A tuple with the rendered value (str) and a flag that indicates whether
        the field widths required adjustment (bool).
        """
        group = self._proc_group(style, adopt=adopt)
        if group == "override":
            # Override the "default" processor key.
            proc_keys = ["width", "override"]
        else:
            # Use the set of processors defined by _setup_fields.
            proc_keys = None

        adjusted = self._set_widths(row, group)
        proc_fields = [self.fields[c](row[c], keys=proc_keys)
                       for c in self.columns]
        return self.style["separator_"].join(proc_fields) + "\n", adjusted