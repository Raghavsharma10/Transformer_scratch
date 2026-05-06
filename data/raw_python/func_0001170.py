def _set_widths(self, row, proc_group):
        """Update auto-width Fields based on `row`.

        Parameters
        ----------
        row : dict
        proc_group : {'default', 'override'}
            Whether to consider 'default' or 'override' key for pre- and
            post-format processors.

        Returns
        -------
        True if any widths required adjustment.
        """
        width_free = self.style["width_"] - sum(
            [sum(self.fields[c].width for c in self.columns),
             self.width_separtor])

        if width_free < 0:
            width_fixed = sum(
                [sum(self.fields[c].width for c in self.columns
                     if c not in self.autowidth_columns),
                 self.width_separtor])
            assert width_fixed > self.style["width_"], "bug in width logic"
            raise elements.StyleError(
                "Fixed widths specified in style exceed total width")
        elif width_free == 0:
            lgr.debug("Not checking widths; no free width left")
            return False

        lgr.debug("Checking width for row %r", row)
        adjusted = False
        for column in sorted(self.columns, key=lambda c: self.fields[c].width):
            # ^ Sorting the columns by increasing widths isn't necessary; we do
            # it so that columns that already take up more of the screen don't
            # continue to grow and use up free width before smaller columns
            # have a chance to claim some.
            if width_free < 1:
                lgr.debug("Giving up on checking widths; no free width left")
                break

            if column in self.autowidth_columns:
                field = self.fields[column]
                lgr.debug("Checking width of column %r "
                          "(field width: %d, free width: %d)",
                          column, field.width, width_free)
                # If we've added any style transform functions as
                # pre-format processors, we want to measure the width
                # of their result rather than the raw value.
                if field.pre[proc_group]:
                    value = field(row[column], keys=[proc_group],
                                  exclude_post=True)
                else:
                    value = row[column]
                value = six.text_type(value)
                value_width = len(value)
                wmax = self.autowidth_columns[column]["max"]
                if value_width > field.width:
                    width_old = field.width
                    width_available = width_free + field.width
                    width_new = min(value_width,
                                    wmax or width_available,
                                    width_available)
                    if width_new > width_old:
                        adjusted = True
                        field.width = width_new
                        lgr.debug("Adjusting width of %r column from %d to %d "
                                  "to accommodate value %r",
                                  column, width_old, field.width, value)
                        self._truncaters[column].length = field.width
                        width_free -= field.width - width_old
                        lgr.debug("Free width is %d after processing column %r",
                                  width_free, column)
        return adjusted