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
        out.append(self._to_str(self.city))
        out.append(self._to_str(self.state_province_region))
        out.append(self._to_str(self.country))
        out.append(self._to_str(self.source))
        out.append(self._to_str(self.wmo))
        out.append(self._to_str(self.latitude))
        out.append(self._to_str(self.longitude))
        out.append(self._to_str(self.timezone))
        out.append(self._to_str(self.elevation))
        return ",".join(out)