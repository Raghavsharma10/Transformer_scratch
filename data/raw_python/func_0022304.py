def export(self, top=True):
        """Exports object to its string representation.

        Args:
            top (bool):  if True appends `internal_name` before values.
                All non list objects should be exported with value top=True,
                all list objects, that are embedded in as fields inlist objects
                should be exported with `top`=False

        Returns:
            str: The objects string representation

        """
        out = []
        if top:
            out.append(self._internal_name)
        out.append(self._to_str(self.number_of_records_per_hour))
        out.append(self._to_str(self.data_period_name_or_description))
        out.append(self._to_str(self.data_period_start_day_of_week))
        out.append(self._to_str(self.data_period_start_day))
        out.append(self._to_str(self.data_period_end_day))
        return ",".join(out)